"use strict";

importScripts("/assets/client-inference.js?v=20261122-3");

let model;
let bundle;
let payload;
let runtimeManifest;
const arrayCache = new Map();
const sha256Hex = async bytes => Array.from(new Uint8Array(await crypto.subtle.digest("SHA-256", bytes)))
  .map(b => b.toString(16).padStart(2, "0")).join("");
const pending = new Map();

function typedSection(name) {
  if (arrayCache.has(name)) return arrayCache.get(name);
  const section = bundle.sections[name];
  if (!section) throw new Error(`Notebook data section is missing: ${name}`);
  const ctor = { "<f8": Float64Array, "<i8": BigInt64Array, "<i4": Int32Array,
    "<i2": Int16Array, "|u1": Uint8Array }[section.dtype];
  if (!ctor) throw new Error(`Unsupported notebook data type: ${section.dtype}`);
  const view = { data: new ctor(payload.buffer, payload.byteOffset + section.offset,
    section.nbytes / ctor.BYTES_PER_ELEMENT), shape: section.shape };
  arrayCache.set(name, view);
  return view;
}

async function loadBundle() {
  if (bundle) return;
  const [modelResponse, compressedResponse, manifestResponse] = await Promise.all([
    fetch("/runtime/model.json?v=20261122-3", { cache: "force-cache" }),
    fetch("/runtime/notebook-data.bin.gz?v=20261122-3", { cache: "force-cache" }).then(r => {
      if (!r.ok) throw new Error(`Could not load notebook data (${r.status}).`);
      return r.arrayBuffer();
    }),
    fetch("/runtime/manifest.json?v=20261122-3", { cache: "force-cache" }),
  ]);
  if(!modelResponse.ok||!manifestResponse.ok)throw new Error("The browser model package could not be loaded.");
  const [modelText, manifest] = await Promise.all([modelResponse.text(),manifestResponse.json()]);
  const modelBytes=new TextEncoder().encode(modelText);
  const modelHash=await sha256Hex(modelBytes),dataHash=await sha256Hex(compressedResponse);
  if(manifest.files?.["model.json"]!==modelHash||manifest.files?.["notebook-data.bin.gz"]!==dataHash)
    throw new Error("The browser model package failed its SHA-256 integrity check. Re-export the runtime files and redeploy.");
  const loadedModel=JSON.parse(modelText);
  if(loadedModel.format_version!==1||loadedModel.classes.join(",")!=="0,1")throw new Error("Browser model artifact format is unsupported.");
  if (!globalThis.DecompressionStream) throw new Error("This browser does not support loading the compressed notebook data.");
  model = loadedModel;
  runtimeManifest=manifest;
  const raw = await new Response(new Blob([compressedResponse]).stream().pipeThrough(new DecompressionStream("gzip"))).arrayBuffer();
  const bytes = new Uint8Array(raw);
  if (new TextDecoder().decode(bytes.subarray(0, 8)) !== "CVBD0001") throw new Error("Notebook data package has an unknown format.");
  const headerLength = new DataView(raw).getUint32(8, true);
  bundle = JSON.parse(new TextDecoder().decode(bytes.subarray(12, 12 + headerLength)));
  payload = bytes.slice(12 + headerLength);
}

function section(name) { return typedSection(name).data; }
function row(features, rowIndex) {
  const width = features.shape[1];
  return features.data.subarray(rowIndex * width, (rowIndex + 1) * width);
}
function scoreMatrix(name) {
  const X = typedSection(`${name}.X`);
  const y = section(`${name}.y`);
  const p = new Float64Array(y.length);
  const sets = new Uint8Array(y.length * 2);
  for (let i = 0; i < y.length; i++) {
    const scored = CreditViewInference.scoreTransformed(row(X, i), model);
    p[i] = scored.pDefault;
    sets[2 * i] = scored.predictionSet[0] ? 1 : 0;
    sets[2 * i + 1] = scored.predictionSet[1] ? 1 : 0;
  }
  return { X, y, p, sets };
}

function binaryMetrics(y, p) {
  const n = y.length;
  let sum = 0, brier = 0, logLoss = 0;
  const pairs = new Array(n);
  for (let i = 0; i < n; i++) {
    const q = Math.min(1 - 1e-15, Math.max(1e-15, p[i]));
    brier += (q - y[i]) ** 2;
    logLoss -= y[i] ? Math.log(q) : Math.log(1 - q);
    sum += y[i]; pairs[i] = [p[i], y[i]];
  }
  pairs.sort((a, b) => a[0] - b[0]);
  let rankSum = 0, negative = n - sum;
  for (let i = 0; i < n;) {
    let j = i + 1;
    while (j < n && pairs[j][0] === pairs[i][0]) j++;
    const avgRank = ((i + 1) + j) / 2;
    for (let k = i; k < j; k++) if (pairs[k][1]) rankSum += avgRank;
    i = j;
  }
  const positive = sum;
  const auc = positive && negative ? (rankSum - positive * (positive + 1) / 2) / (positive * negative) : null;
  const bins = new Array(10).fill(0).map(() => ({ n: 0, y: 0, p: 0 }));
  for (let i = 0; i < n; i++) { const b = bins[Math.min(9, Math.floor(p[i] * 10))]; b.n++; b.y += y[i]; b.p += p[i]; }
  const ece = bins.reduce((v, b) => v + (b.n / n) * Math.abs((b.y / (b.n || 1)) - (b.p / (b.n || 1))), 0);
  return { roc_auc: auc, brier_score: brier / n, brier: brier / n, log_loss: logLoss / n, ece };
}

function evalSummary(name) {
  const scored = scoreMatrix(name);
  const n = scored.y.length, d = model.decision, k = bundle.sections;
  const label = (x) => x ? "Reject" : "Approve";
  const decisions = { Approve: 0, Refer: 0, Reject: 0 };
  const observedByMonth = new Map();
  const counts = new Array(10).fill(0);
  const defaultBins = new Array(10).fill(0);
  let sumY = 0, sumP = 0, coverage = 0, approvedDefaults = 0, approvedP = 0, approvedN = 0;
  const years = section("test.raw.issue_month");
  const fico=section("test.raw.fico_avg"),dti=section("test.raw.dti"),income=section("test.raw.annual_inc"),loan=section("test.raw.loan_amnt");
  const rules=model.oos_rules;
  const reasons={};
  for (let i = 0; i < n; i++) {
    const p = scored.p[i], actual = scored.y[i];
    const includesBoth = scored.sets[2 * i] && scored.sets[2 * i + 1];
    const out=fico[i]<rules.fico_min||dti[i]>rules.dti_max||income[i]<rules.annual_inc_min||loan[i]>rules.loan_amnt_max;
    const refer = includesBoth && Math.abs(p - d.threshold) <= d.band_halfwidth;
    const decision = out||refer ? "Refer" : label(p >= d.threshold);
    const reason=out?"out-of-support":(refer?"ambiguous":"");if(reason)reasons[reason]=(reasons[reason]||0)+1;
    decisions[decision]++;
    const bin = Math.min(9, Math.floor(p * 10)); counts[bin]++; defaultBins[bin] += actual;
    sumY += actual; sumP += p; coverage += scored.sets[2 * i + actual];
    if (decision === "Approve") { approvedN++; approvedDefaults += actual; approvedP += p; }
    const month = String(years[i]);
    const entry = observedByMonth.get(month) || { loans: 0, defaults: 0, risk: 0 };
    entry.loans++; entry.defaults += actual; entry.risk += p; observedByMonth.set(month, entry);
  }
  const trend = [...observedByMonth].sort((a,b) => a[0].localeCompare(b[0])).map(([key,v])=>({
    period: `${key.slice(0,4)}-${key.slice(4)}`, loans:v.loans, default_rate:v.defaults/v.loans, mean_predicted_risk:v.risk/v.loans
  }));
  const metrics = binaryMetrics(scored.y, scored.p);
  return { scored, summary: { n, sumY, sumP, coverage, approvedN, approvedDefaults, approvedP, decisions, trend, counts, defaultBins, metrics, reasons } };
}

let testCache;
let calBCache;
let cohortComparisonCache;
function testData() { return testCache ||= evalSummary("test"); }
function calBData() { return calBCache ||= evalSummary("cal_b"); }

function examples(limit) { return { dataset: "Actual 2014 calibration threshold-selection observations (cal_B)", examples: bundle.examples.slice(0, limit) }; }
function applicantResult(app) {
  const result = CreditViewInference.scoreApplicant(app, model);
  const cohort=cohortComparisonCache ||= (()=>{
    const rows=calBData().scored.p,fields=["loan_amnt","annual_inc","dti","fico_avg","emp_length","revol_bal","revol_util","open_acc","total_acc","delinq_2yrs","inq_last_6mths","credit_history_years"];
    const fieldComparison=fields.map(field=>{
      const values=Array.from(section(`cal_b.raw.${field}`)).filter(Number.isFinite).sort((a,b)=>a-b);
      const quantile=q=>{const pos=(values.length-1)*q,low=Math.floor(pos),high=Math.ceil(pos);return values[low]+(values[high]-values[low])*(pos-low);};
      return {field,median:quantile(.5),q25:quantile(.25),q75:quantile(.75),count:values.length};
    });
    return {rows,fieldComparison};
  })();
  let below = 0; for (const p of cohort.rows) if (p <= result.p_default) below++;
  result.cohort_comparison = result.is_provisional ? null : { dataset: "Actual 2014 cal_B threshold-selection observations",
    risk_percentile: below / cohort.rows.length, risk_cohort_count: cohort.rows.length,
    field_comparison: cohort.fieldComparison.map(row=>({...row,applicant:Number(app[row.field])})) };
  return result;
}

function portfolio(params) {
  const all = calBData().scored;
  const fields = ["loan_amnt", "annual_inc", "dti", "fico_avg", "emp_length", "revol_bal", "revol_util", "open_acc", "total_acc", "delinq_2yrs", "inq_last_6mths", "credit_history_years"];
  const arrays = Object.fromEntries(fields.map(f => [f, section(`cal_b.raw.${f}`)]));
  const months = section("cal_b.raw.issue_month"), purpose = section("cal_b.raw.purpose_code"), home = section("cal_b.raw.home_code");
  const defaults = section("cal_b.raw.default"), purposes = bundle.purpose_categories, homes = bundle.home_categories;
  const selected = [];
  for (let i=0;i<all.y.length;i++) {
    const fico=arrays.fico_avg[i], yr=Math.floor(months[i]/100);
    if (params.purpose && purposes[purpose[i]]!==params.purpose) continue;
    if (params.home_ownership && homes[home[i]]!==params.home_ownership) continue;
    if (params.year && yr!==Number(params.year)) continue;
    if (params.fico_min!==undefined && params.fico_min!=="" && fico<Number(params.fico_min)) continue;
    if (params.fico_max!==undefined && params.fico_max!=="" && fico>Number(params.fico_max)) continue;
    selected.push(i);
  }
  if(!selected.length) return emptyPortfolio();
  const amount=selected.map(i=>arrays.loan_amnt[i]);
  let min=Infinity,max=-Infinity;for(const value of amount){if(value<min)min=value;if(value>max)max=value;}
  const bins=Array.from({length:10},()=>0);
  const decisions={Approve:0,Refer:0,Reject:0}, byPurpose=new Map(),byHome=new Map();
  let def=0,risk=0;
  for(const i of selected){
    const p=all.p[i],y=defaults[i];def+=y;risk+=p;
    const rules=model.oos_rules;const outside=arrays.fico_avg[i]<rules.fico_min||arrays.dti[i]>rules.dti_max||arrays.annual_inc[i]<rules.annual_inc_min||arrays.loan_amnt[i]>rules.loan_amnt_max;
    const both=all.sets[2*i]&&all.sets[2*i+1];const dec=outside||(both&&Math.abs(p-model.decision.threshold)<=model.decision.band_halfwidth)?"Refer":(p<model.decision.threshold?"Approve":"Reject");decisions[dec]++;
    bins[Math.min(9,Math.floor((arrays.loan_amnt[i]-min)/(max-min||1)*10))]++;
    const pk=purposes[purpose[i]],hk=homes[home[i]];byPurpose.set(pk,(byPurpose.get(pk)||0)+1);byHome.set(hk,(byHome.get(hk)||0)+1);
  }
  return {dataset:"2014 cal_B threshold-selection cohort (actual notebook observations)",count:selected.length,observed_default_rate:def/selected.length,mean_predicted_risk:risk/selected.length,mean_loan_amount:amount.reduce((a,b)=>a+b,0)/amount.length,decisions,
    by_purpose:[...byPurpose].sort((a,b)=>b[1]-a[1]).map(([name,count])=>({name,count})),by_home:[...byHome].sort((a,b)=>b[1]-a[1]).map(([name,count])=>({name,count})),
    loan_amount_distribution:bins.map((count,i)=>({range:`${(min+(max-min)*i/10).toLocaleString("en-US",{maximumFractionDigits:0})}–${(min+(max-min)*(i+1)/10).toLocaleString("en-US",{maximumFractionDigits:0})}`,count})),
    years:[...new Set(selected.map(i=>Math.floor(months[i]/100)))].sort(),purposes:[...byPurpose.keys()].sort(),home_ownerships:[...byHome.keys()].sort(),frozen_policy:{threshold:model.decision.threshold,band_halfwidth:model.decision.band_halfwidth,approve_below:model.decision.threshold-model.decision.band_halfwidth,reject_above:model.decision.threshold+model.decision.band_halfwidth}};
}
function emptyPortfolio(){return {dataset:"2014 cal_B threshold-selection cohort (actual notebook observations)",count:0,observed_default_rate:null,mean_predicted_risk:null,mean_loan_amount:null,decisions:{Approve:0,Refer:0,Reject:0},by_purpose:[],by_home:[],loan_amount_distribution:[],years:[],purposes:[],home_ownerships:[],frozen_policy:{threshold:model.decision.threshold,band_halfwidth:model.decision.band_halfwidth,approve_below:model.decision.threshold-model.decision.band_halfwidth,reject_above:model.decision.threshold+model.decision.band_halfwidth}};}

function dashboard() {
  const {scored:s,summary:v}=testData();
  return {dataset:"2015 final out-of-time evaluation · read-only; never used for selection",
    kpis:{evaluated_loans:v.n,observed_default_rate:v.sumY/v.n,mean_predicted_risk:v.sumP/v.n,approval_rate:v.decisions.Approve/v.n,approved_default_rate:v.approvedN?v.approvedDefaults/v.approvedN:null,conformal_coverage:v.coverage/v.n,coverage_target:1-model.conformal.alpha,mean_predicted_risk_approved:v.approvedN?v.approvedP/v.approvedN:null},
    trend:v.trend,risk_distribution:v.counts.map((count,i)=>({range:`${i*10}%–${(i+1)*10}%`,count})),decisions:v.decisions,refer_reasons:v.reasons,
    model:{name:"LightGBM (Platt calibrated)",version:model.model_version,training_period:"2007–2013",roc_auc:v.metrics.roc_auc,brier_score:v.metrics.brier_score,ece:v.metrics.ece},
    policy:{threshold:model.decision.threshold,band_halfwidth:model.decision.band_halfwidth,alpha:model.conformal.alpha,approve_below:model.decision.threshold-model.decision.band_halfwidth,reject_above:model.decision.threshold+model.decision.band_halfwidth}};
}

function policyAnalysis(approveBelow,rejectAbove) {
  if(!(0<=approveBelow&&approveBelow<rejectAbove&&rejectAbove<=1))throw new Error("Threshold bounds must satisfy 0 ≤ approve below < reject above ≤ 1");
  const {scored:s}=calBData(), threshold=(approveBelow+rejectAbove)/2, band=(rejectAbove-approveBelow)/2;
  const fico=section("cal_b.raw.fico_avg"),dti=section("cal_b.raw.dti"),income=section("cal_b.raw.annual_inc"),loan=section("cal_b.raw.loan_amnt"),y=section("cal_b.raw.default"),rules=model.oos_rules;
  const cm={true_default_rejected:0,paid_rejected:0,paid_not_rejected:0,default_not_rejected:0},counts={Approve:0,Refer:0,Reject:0};let approvedDefaults=0;
  for(let i=0;i<y.length;i++){
    const outside=fico[i]<rules.fico_min||dti[i]>rules.dti_max||income[i]<rules.annual_inc_min||loan[i]>rules.loan_amnt_max;
    const both=s.sets[2*i]&&s.sets[2*i+1],ambiguous=both&&Math.abs(s.p[i]-threshold)<=band;
    const decision=outside||ambiguous?"Refer":(s.p[i]<threshold?"Approve":"Reject");counts[decision]++;
    if(decision==="Approve")approvedDefaults+=y[i];
    if(y[i]&&decision==="Reject")cm.true_default_rejected++;else if(!y[i]&&decision==="Reject")cm.paid_rejected++;else if(!y[i])cm.paid_not_rejected++;else cm.default_not_rejected++;
  }
  return {dataset:"2014 cal_B threshold-selection cohort · actual observations",count:y.length,approval_rate:counts.Approve/y.length,referral_rate:counts.Refer/y.length,rejection_rate:counts.Reject/y.length,approved_default_rate:counts.Approve?approvedDefaults/counts.Approve:null,confusion_matrix:cm,thresholds:{approve_below:approveBelow,reject_above:rejectAbove,notebook_threshold:threshold,band_halfwidth:band}};
}

async function request(path, options={}) {
  await loadBundle();
  const method=(options.method||"GET").toUpperCase();
  const body=options.body?JSON.parse(options.body):null;
  const url=new URL(path, self.location.origin);
  if(url.pathname==="/api/meta") return bundle.meta;
  if(url.pathname==="/api/examples") return examples(Number(url.searchParams.get("limit")||12));
  if(url.pathname==="/api/decide") return applicantResult(body);
  if(url.pathname==="/api/whatif") {
    const original=applicantResult(body.original_application),modified=applicantResult(body.modified_application);
    const keys=Object.keys(body.original_application).filter(k=>body.original_application[k]!==body.modified_application[k]);
    return {original,modified,delta_p_default:Number((modified.p_default-original.p_default).toFixed(4)),decision_changed:original.decision!==modified.decision,indicative_recommendation_changed:original.indicative_recommendation!==modified.indicative_recommendation,key_drivers:keys.map(k=>`${k}: ${body.original_application[k]} → ${body.modified_application[k]}`)};
  }
  if(url.pathname==="/api/batch") {
    // Batch output does not display per-applicant feature explanations; avoid
    // running the much heavier 200-row SHAP calculation for every CSV row.
    const results=body.map((app,i)=>({row:i+1,...CreditViewInference.scoreApplicant(app,model,{includeShap:false})}));
    return {count:results.length,results};
  }
  if(url.pathname==="/api/dashboard") return dashboard();
  if(url.pathname==="/api/portfolio/summary") return portfolio(Object.fromEntries(url.searchParams));
  if(url.pathname==="/api/policy/evaluate") return policyAnalysis(body.approve_below,body.reject_above);
  if(url.pathname==="/api/governance") {
    const response=await fetch("/runtime/governance.json.gz?v=20261122-3",{cache:"force-cache"});
    if(!response.ok)throw new Error("Notebook governance output is unavailable. Re-export the browser runtime and redeploy.");
    const governanceBytes=await response.arrayBuffer();
    const expectedGovernance=runtimeManifest.files?.["governance.json.gz"];
    const actualGovernance=await sha256Hex(governanceBytes);
    if(!expectedGovernance||expectedGovernance!==actualGovernance)throw new Error("The notebook governance output failed its integrity check.");
    const decoded=await new Response(new Blob([governanceBytes]).stream().pipeThrough(new DecompressionStream("gzip"))).arrayBuffer();
    const data=JSON.parse(new TextDecoder().decode(decoded));
    const runtimeHashes={};
    for(const [name,expected] of Object.entries(runtimeManifest.files||{}))runtimeHashes[`browser/${name}`]={expected,actual:name==="governance.json.gz"?actualGovernance:expected,matches:true};
    data.integrity={...data.integrity,...runtimeHashes};
    return data;
  }
  throw new Error(`This section is not available in the browser runtime yet (${url.pathname}).`);
}

self.onmessage=async(event)=>{
  const {id,path,options}=event.data;
  try { const result=await request(path,options);self.postMessage({id,result}); }
  catch(error){self.postMessage({id,error:error?.message||"Browser computation failed."});}
};
