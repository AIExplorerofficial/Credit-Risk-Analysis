"use strict";
const $ = (s, root = document) => root.querySelector(s);
const $$ = (s, root = document) => [...root.querySelectorAll(s)];
const state = { meta: null, examples: [], dashboard: null, portfolio: null, assessment: null, scenarioApplicant: null, governance: null };
const fieldGroups = [
  { title: "Loan information", fields: ["loan_amnt", "purpose"] },
  { title: "Personal finances", fields: ["annual_inc", "emp_length", "home_ownership"] },
  { title: "Credit profile", fields: ["fico_avg", "credit_history_years", "open_acc", "total_acc"] },
  { title: "Utilisation & obligations", fields: ["dti", "revol_bal", "revol_util", "delinq_2yrs", "inq_last_6mths", "pub_rec"] },
];
const labels = { loan_amnt: "Requested loan amount ($)", annual_inc: "Annual income ($)", dti: "Debt-to-income ratio (%)", fico_avg: "FICO score", emp_length: "Employment length (years)", home_ownership: "Home ownership", purpose: "Loan purpose", open_acc: "Open credit accounts", pub_rec: "Public records", revol_bal: "Revolving balance ($)", revol_util: "Revolving utilization (%)", total_acc: "Total credit accounts", delinq_2yrs: "Delinquencies in past 2 years", inq_last_6mths: "Credit inquiries in past 6 months", credit_history_years: "Credit history (years)" };
const fieldMeanings = { loan_amnt:"Principal requested, in US dollars.", annual_inc:"Annual income reported for the applicant; enter dollars per year.", dti:"Debt compared with income, expressed as a percentage (for example, enter 25 for 25%).", fico_avg:"FICO score used by the notebook model; typical values range from 300 to 850.", emp_length:"Years in employment. The training data records years, with under one year treated as zero.", home_ownership:"Housing category recorded in the historical dataset.", purpose:"Stated reason for the loan; choose a category observed in the notebook data.", open_acc:"Number of currently open credit accounts.", pub_rec:"Count of derogatory public records in the historical application data.", revol_bal:"Outstanding revolving credit balance, in US dollars.", revol_util:"Share of revolving credit limit in use, as a percentage (enter 30 for 30%).", total_acc:"Number of credit accounts recorded across the applicant’s history.", delinq_2yrs:"Recorded 30-day-or-longer payment delinquencies during the prior two years.", inq_last_6mths:"Credit inquiries recorded during the prior six months.", credit_history_years:"Years since the earliest reported credit line." };
const featureShortLabels = { loan_amnt:"Loan amount", annual_inc:"Annual income", dti:"Debt / income", fico_avg:"FICO score", emp_length:"Employment", home_ownership:"Home status", purpose:"Loan purpose", open_acc:"Open accounts", pub_rec:"Public records", revol_bal:"Revolving debt", revol_util:"Credit utilization", total_acc:"Total accounts", delinq_2yrs:"Past delinquencies", inq_last_6mths:"Recent inquiries", credit_history_years:"Credit history" };
const escapeHtml = value => String(value ?? "").replace(/[&<>"']/g, c => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
const fmt = (value, digits = 1) => value == null || !Number.isFinite(Number(value)) ? "—" : Number(value).toLocaleString(undefined, { maximumFractionDigits: digits });
const pct = (value, digits = 1) => value == null || !Number.isFinite(Number(value)) ? "—" : `${(Number(value) * 100).toFixed(digits)}%`;
const money = value => value == null || !Number.isFinite(Number(value)) ? "—" : Number(value).toLocaleString(undefined, { style: "currency", currency: "USD", maximumFractionDigits: 0 });
async function api(path, options = {}) {
  const controller = new AbortController();
  const timeout = window.setTimeout(() => controller.abort(), 45000);
  let response;
  try {
    response = await fetch(path, { headers: { "Content-Type": "application/json", ...(options.headers || {}) }, ...options, signal: controller.signal });
  } catch (error) {
    if (error.name === "AbortError") throw new Error("This request took too long. Check the connection and try again.");
    throw new Error("CreditView could not reach the local service. Check that it is running and retry.");
  } finally { window.clearTimeout(timeout); }
  if (!response.ok) {
    let message = `Request failed (${response.status})`;
    try { const data = await response.json(); message = typeof data.detail === "string" ? data.detail : JSON.stringify(data.detail || data); } catch {}
    throw new Error(message);
  }
  return response.json();
}
let activeOperations = 0;
function busy(on) { activeOperations = Math.max(0, activeOperations + (on ? 1 : -1)); const active = activeOperations > 0; $("#loading").classList.toggle("hidden", !active); document.body.classList.toggle("request-active", active); }
function lockAction(button, action) {
  button.addEventListener("click", async event => {
    if (button.disabled) return;
    button.disabled = true;
    try { await action(event); } finally { button.disabled = false; }
  });
}
function transientError(error) { return /could not reach|too long|request failed \(5\d\d\)/i.test(error.message || ""); }
function addRetry(container, action) { const button=document.createElement("button");button.type="button";button.className="text-button retry-button";button.textContent="Retry";button.addEventListener("click",async()=>{button.disabled=true;try{await action();}finally{button.remove();}},{once:true});container.append(" ",button); }
function notice(text, type = "info", retry = null) {
  const el = $("#notice"); el.replaceChildren(); el.dataset.state = type;
  const message = document.createElement("span"); message.textContent = text; el.append(message);
  if (retry) { const button = document.createElement("button"); button.className = "text-button retry-button"; button.textContent = "Retry"; button.addEventListener("click", retry, { once: true }); el.append(button); }
  el.classList.remove("hidden"); window.clearTimeout(notice.timer);
  notice.timer = window.setTimeout(() => el.classList.add("hidden"), retry ? 15000 : 6000);
}
function showPage(name) {
  $$(".page").forEach(p => p.classList.toggle("active", p.id === `page-${name}`));
  $$(".nav-link").forEach(b => b.classList.toggle("active", b.dataset.page === name));
  $(".nav").classList.remove("open"); history.replaceState(null, "", `#${name}`);
  if (name === "portfolio" && !state.portfolio) loadPortfolio();
  if (name === "governance" && !state.governance) loadGovernance();
}
function fieldMarkup(field, prefix = "app") {
  const id = `${prefix}-${field.name}`;
  const label = labels[field.name] || field.label || field.name;
  const attrs = [field.min != null ? `min="${field.min}"` : "", field.max != null ? `max="${field.max}"` : "", field.step != null ? `step="${field.step}"` : ""].filter(Boolean).join(" ");
  let input;
  if (field.type === "select") input = `<select class="field-input" id="${id}" name="${field.name}" required><option value="" selected disabled>Select…</option>${(field.options || []).map(v => `<option value="${escapeHtml(v)}">${escapeHtml(v.replaceAll("_", " "))}</option>`).join("")}</select>`;
  else input = `<input class="field-input" id="${id}" name="${field.name}" type="number" ${attrs} required inputmode="decimal">`;
  return `<label class="field" for="${id}"><span>${escapeHtml(label)}</span>${input}<small class="field-help">${escapeHtml(fieldMeanings[field.name]||"This value is passed through the notebook feature-processing steps.")}</small></label>`;
}
function renderFieldForm(target, prefix, compact = false) {
  const fields = new Map(state.meta.fields.map(f => [f.name, f]));
  if (!compact) {
    $(target).innerHTML = fieldGroups.map(group => `<section class="field-group"><h3><i class="group-dot"></i>${group.title}</h3><div class="field-grid">${group.fields.map(k => fieldMarkup(fields.get(k), prefix)).join("")}</div></section>`).join("");
  } else {
    $(target).innerHTML = fieldGroups.map(group => `<section class="field-group"><h3><i class="group-dot"></i>${group.title}</h3><div class="field-grid">${group.fields.map(k => fieldMarkup(fields.get(k), prefix)).join("")}</div></section>`).join("");
  }
}
function renderBatchFieldGuide() {
  const container = $("#batchFieldList");
  if (!container || !state.meta?.fields?.length) return;
  const rows = state.meta.fields.map(field => {
    const accepted = field.type === "select"
      ? (field.options || []).map(escapeHtml).join(", ") || "No category values available"
      : "Numeric value";
    return `<tr><td><code>${escapeHtml(field.name)}</code></td><td>${escapeHtml(field.label || labels[field.name] || field.name)}<br><small>${escapeHtml(fieldMeanings[field.name] || "")}</small></td><td>${accepted}</td></tr>`;
  }).join("");
  container.innerHTML = `<table class="data-table batch-fields-table"><thead><tr><th>CSV column name</th><th>What it represents</th><th>Accepted value format</th></tr></thead><tbody>${rows}</tbody></table>`;
}
function downloadBatchTemplate() {
  const fields = state.meta?.fields || [];
  if (!fields.length) { notice("The required fields are not available yet. Reload the page and try again.", "error"); return; }
  const header = fields.map(field => `"${String(field.name).replace(/"/g, '""')}"`).join(",");
  download("creditview-batch-template.csv", `${header}\r\n`, "text/csv;charset=utf-8");
}
function readForm(prefix, names = state.meta.fields.map(f => f.name)) {
  const result = {}; const missing = []; const invalid = [];
  for (const name of names) {
    const el = $(`#${prefix}-${name}`);
    if (!el || el.value === "") { missing.push(labels[name] || name); continue; }
    const field = state.meta.fields.find(f => f.name === name);
    if (field.type === "select") {
      if (!field.options?.includes(el.value)) invalid.push(labels[name] || name); else result[name] = el.value;
    } else {
      const value = Number(el.value);
      if (!Number.isFinite(value)) missing.push(labels[name] || name);
      else if ((field.min != null && (field.exclusive_min ? value <= field.min : value < field.min)) || (field.max != null && (field.exclusive_max ? value >= field.max : value > field.max))) invalid.push(labels[name] || name);
      else result[name] = value;
    }
  }
  if (missing.length) throw new Error(`Please provide: ${missing.join(", ")}.`);
  if (invalid.length) throw new Error(`Check the allowed values for: ${invalid.join(", ")}.`);
  return result;
}
function putForm(prefix, applicant) {
  for (const [key, value] of Object.entries(applicant)) {
    const el = $(`#${prefix}-${key}`);
    if (el) el.value = value ?? "";
  }
}
function renderKpi(item) { return `<article class="kpi"><div class="kpi-top"><span class="kpi-label">${escapeHtml(item.label)}</span><span class="kpi-icon">${item.icon || "◈"}</span></div><div class="kpi-value">${escapeHtml(item.value)}</div><div class="kpi-note">${escapeHtml(item.note)}</div>${item.meaning?`<p class="kpi-meaning">${escapeHtml(item.meaning)}</p>`:""}</article>`; }
function drawTrend(el, rows) {
  if (!rows?.length) { el.innerHTML = `<div class="empty-state">No trend data is available.</div>`; return; }
  const w = 700, h = 215, pad = { l: 36, r: 12, t: 12, b: 31 }, xs = rows.map((_, i) => pad.l + i * (w - pad.l - pad.r) / Math.max(1, rows.length - 1));
  const all = rows.flatMap(d => [d.default_rate, d.mean_predicted_risk]); const max = Math.max(.01, ...all) * 1.15;
  const y = v => h - pad.b - (v / max) * (h - pad.t - pad.b);
  const observed = rows.map((d, i) => `${i ? "L" : "M"}${xs[i].toFixed(1)},${y(d.default_rate).toFixed(1)}`).join(" ");
  const predicted = rows.map((d, i) => `${i ? "L" : "M"}${xs[i].toFixed(1)},${y(d.mean_predicted_risk).toFixed(1)}`).join(" ");
  const area = `${observed} L${xs.at(-1)},${h-pad.b} L${xs[0]},${h-pad.b} Z`;
  const grid = [0,.25,.5,.75,1].map(f => { const yy=pad.t+f*(h-pad.t-pad.b);return `<line class="chart-grid" x1="${pad.l}" y1="${yy}" x2="${w-pad.r}" y2="${yy}"/><text class="chart-label" x="0" y="${yy+3}">${((1-f)*max*100).toFixed(0)}%</text>`; }).join("");
  const step = Math.max(1, Math.floor(rows.length/5)); const ticks=rows.filter((_,i)=>i%step===0||i===rows.length-1).map(d=>{const i=rows.indexOf(d);return `<text class="chart-label" text-anchor="middle" x="${xs[i]}" y="${h-8}">${escapeHtml(d.period.slice(5))}</text>`;}).join("");
  const title = rows.map(d => `${d.period}: defaults ${pct(d.default_rate,2)} · model ${pct(d.mean_predicted_risk,2)} · ${fmt(d.loans,0)} loans`).join("&#10;");
  el.innerHTML = `<svg viewBox="0 0 ${w} ${h}" role="img" aria-label="Monthly default and predicted risk trend">${grid}<defs><linearGradient id="chartFill" x1="0" x2="0" y1="0" y2="1"><stop offset="0%" stop-color="var(--accent)"/><stop offset="100%" stop-color="var(--surface)" stop-opacity=".05"/></linearGradient></defs><path class="chart-area" d="${area}"/><path class="chart-line observed" d="${observed}"/><path class="chart-line secondary" d="${predicted}"/>${xs.map((x,i)=>`<circle class="chart-dot" cx="${x}" cy="${y(rows[i].default_rate)}" r="3"><title>${escapeHtml(rows[i].period)} · observed ${pct(rows[i].default_rate,2)} · predicted ${pct(rows[i].mean_predicted_risk,2)} · ${fmt(rows[i].loans,0)} loans</title></circle>`).join("")}${ticks}<title>${title}</title></svg>`;
}
function drawXYChart(el, rows, xKey, yKey, identity = false, aria = "Model evaluation curve", xLabel = xKey, yLabel = yKey) {
  if (!rows?.length) { el.textContent = "No curve data is available."; return; }
  const w=600,h=215,p={l:42,r:15,t:12,b:32},x=v=>p.l+Math.max(0,Math.min(1,Number(v)))*(w-p.l-p.r),y=v=>h-p.b-Math.max(0,Math.min(1,Number(v)))*(h-p.t-p.b);
  const grid=[0,.25,.5,.75,1].map(f=>`<line class="chart-grid" x1="${x(0)}" y1="${y(f)}" x2="${x(1)}" y2="${y(f)}"/><text class="chart-label" x="3" y="${y(f)+3}">${f.toFixed(2)}</text>`).join("");
  const path=rows.map((r,i)=>`${i?"L":"M"}${x(r[xKey])},${y(r[yKey])}`).join(" ");
  const reference=identity?`<line x1="${x(0)}" y1="${y(0)}" x2="${x(1)}" y2="${y(1)}" stroke="var(--chart-secondary)" stroke-dasharray="5 5"/>`:"";
  el.innerHTML=`<svg viewBox="0 0 ${w} ${h}" role="img" aria-label="${escapeHtml(aria)}">${grid}${reference}<path class="chart-line" d="${path}"/>${rows.map((r,i)=>i%Math.max(1,Math.floor(rows.length/18))===0?`<circle cx="${x(r[xKey])}" cy="${y(r[yKey])}" r="2.5" fill="var(--chart-line)"><title>${escapeHtml(xLabel)} ${fmt(r[xKey],3)} · ${escapeHtml(yLabel)} ${fmt(r[yKey],3)}</title></circle>`:"").join("")}<text class="chart-label" x="${p.l}" y="${h-8}">${escapeHtml(xLabel)}</text><text class="chart-label" text-anchor="end" x="${w-p.r}" y="${p.t+9}">${escapeHtml(yLabel)}</text></svg>`;
}
function renderBars(el, rows, valueKey = "count", labelKey = "range", color = "") {
  const max = Math.max(1, ...rows.map(r => Number(r[valueKey]) || 0));
  const digits = valueKey === "importance" ? 4 : 0;
  el.innerHTML = rows.length ? rows.map(r => {const raw=String(r[labelKey]);const label=labelKey==="feature"?(labels[raw]||raw.replaceAll("_"," ")):raw;const short=labelKey==="feature"?(featureShortLabels[raw]||label):label;return `<div class="bar-row"><span class="bar-label" title="${escapeHtml(label)}">${escapeHtml(short)}</span><div class="bar-track" title="${escapeHtml(label)}: ${fmt(r[valueKey],digits)}${valueKey==="importance"?" mean absolute SHAP contribution":" observations"}"><div class="bar-fill" style="width:${Math.max(1, (r[valueKey] / max) * 100)}%;${color ? `background:${color}` : ""}"></div></div><span class="bar-count">${fmt(r[valueKey],digits)}</span></div>`;}).join("") : `<p class="muted small">No rows match the selected cohort.</p>`;
}
function renderDecisions(el, counts) {
  const total = Object.values(counts).reduce((a,b)=>a+b,0) || 1;
  el.innerHTML = ["Approve","Refer","Reject"].map(name => `<div class="decision-row"><span class="decision-label"><i class="decision-key ${name.toLowerCase()}"></i>${name}</span><div class="decision-track"><div class="decision-fill ${name.toLowerCase()}" style="width:${(counts[name]||0)/total*100}%"></div></div><span class="decision-count">${fmt(counts[name]||0,0)} · ${pct((counts[name]||0)/total,1)}</span></div>`).join("");
}
async function loadDashboard() {
  try {
    state.dashboard = await api("/api/dashboard"); const d = state.dashboard, k = d.kpis;
    $("#dashboardScope").textContent = d.dataset;
    $("#kpis").innerHTML = [
      {label:"Evaluated loans",value:fmt(k.evaluated_loans,0),note:d.dataset,meaning:"Number of held-out 2015 records included here.",icon:"▤"},
      {label:"Observed default rate",value:pct(k.observed_default_rate,2),note:"Actual outcomes · final OOT",meaning:"Share of these loans recorded as Charged Off; this is a historical outcome.",icon:"⌁"},
      {label:"Mean predicted risk",value:pct(k.mean_predicted_risk,2),note:"Frozen calibrated model · final OOT",meaning:"Average model-estimated chance of default for the same loans.",icon:"◉"},
      {label:"Approval rate",value:pct(k.approval_rate,2),note:`Approved cohort default: ${pct(k.approved_default_rate,2)}`,meaning:"Share marked Approve by the saved rule; the note gives their later observed default share.",icon:"✓"},
    ].map(renderKpi).join("");
    drawTrend($("#trendChart"),d.trend);renderBars($("#riskBars"),d.risk_distribution,"count","range");renderDecisions($("#decisionMix"),d.decisions);
    $("#modelName").textContent=d.model.name;$("#modelVersion").textContent=`${d.model.version} · trained ${d.model.training_period}`;
    $("#snapshotAuc").textContent=fmt(d.model.roc_auc,4);$("#snapshotBrier").textContent=fmt(d.model.brier_score,4);$("#snapshotPolicy").textContent=`${pct(d.policy.threshold,0)} ± ${pct(d.policy.band_halfwidth,0)}`;
    $("#boundaries").textContent=`Saved threshold ${pct(d.policy.threshold,0)} · referral band ±${pct(d.policy.band_halfwidth,0)} only when both outcomes remain plausible · 2015 conformal coverage ${pct(k.conformal_coverage,2)} (target ${pct(k.coverage_target,0)}). Coverage is the share of actual outcomes included in the model’s prediction sets.`;
    $("#integrityCount").textContent="Checksums are available on the governance page";
  } catch(e) { notice(e.message,"error",loadDashboard);$("#dashboardScope").textContent="Dashboard data could not be loaded."; }
}
async function loadExamples() {
  const data=await api(`/api/examples?limit=${state.meta.app_limits.historical_example_count}`);state.examples=data.examples;
  for(const id of ["examplePicker","scenarioExample"]){const el=$(`#${id}`);el.innerHTML=`<option value="">Choose historical example…</option>`+state.examples.map((x,i)=>`<option value="${i}">Record ${i+1} · ${escapeHtml(x.originated)} · ${escapeHtml(x.historical_outcome)}</option>`).join("");}
}
function applicantFromExample(index) { const ex=state.examples[Number(index)]; if(!ex) throw new Error("Choose a historical example first.");return ex; }
function renderAssessmentResult(result, applicant, historic = null) {
  state.assessment={result,applicant}; const formalStatus=result.formal_status||result.decision; const indicative=result.indicative_recommendation||formalStatus; const tone=result.is_provisional?"refer":formalStatus.toLowerCase();
  const factors=(result.top_factors||[]).map(f=>`<div class="factor"><div class="factor-head"><span>${escapeHtml(labels[f.feature]||f.feature.replaceAll("_"," "))}</span><span class="factor-tag ${f.impact==='supportive'?'supportive':''}">${f.impact==='supportive'?'Pulls this score lower':'Pulls this score higher'}</span></div><div class="factor-detail">${escapeHtml(f.explanation)}</div></div>`).join("")||`<p class="muted small">No feature contributions were available for this assessment.</p>`;
  const history = historic ? `<p class="result-disclaimer">Historical notebook cal_B record · ${escapeHtml(historic.originated)} · observed outcome: ${escapeHtml(historic.historical_outcome)}. This outcome is hindsight, not an input to the assessment.</p>` : "";
  const decisionMeaning=result.is_provisional?"The indicative outcome follows the saved model rule but does not replace the required human review.":{approve:"Estimated risk is below the saved notebook threshold. This is a model recommendation, not a lender’s offer.",refer:"The conformal model considers both outcomes plausible near the cutoff, so this case is referred for review.",reject:"Estimated risk is at or above the saved notebook threshold. This is a model recommendation, not a lender’s final decision."}[tone];
  const indicativePanel=`<div class="indicative-panel"><span>Indicative model recommendation</span><strong>${result.is_provisional?'INDICATIVE ':''}${escapeHtml(indicative.toUpperCase())}</strong><small>${result.is_provisional?'This is not the formal assessment and does not replace manual review.':'Matches the formal status under the existing supported-input decision policy.'}</small></div>`;
  const supportCard=result.is_provisional||result.support_warnings?.length?`<article class="card result-card support-warning"><h3>${result.is_provisional?'Why manual review is required':'Input values to review'}</h3><p>${escapeHtml(result.formal_explanation||'Some inputs are outside or unusual compared with the notebook training data.')}</p><ul>${(result.support_warnings||[]).map(w=>`<li>${escapeHtml(w)}</li>`).join('')}</ul>${result.is_provisional?'<p class="section-explainer">Historical cohort comparison is omitted because this estimate is outside model support.</p>':''}</article>`:"";
  $("#assessmentResult").innerHTML=`<article class="result-hero ${tone}"><div class="result-top"><span class="result-label">FORMAL ASSESSMENT</span><span class="tag tag-dark">${result.is_provisional?'OUTSIDE MODEL SUPPORT':'WITHIN MODEL SUPPORT'}</span></div><div class="result-decision">${escapeHtml(formalStatus)}</div>${indicativePanel}<div class="result-prob">${pct(result.p_default,2)}</div><div class="result-desc">${result.is_provisional?'Estimated default risk · PROVISIONAL and potentially unreliable for this application.':'Estimated chance of default'} · saved decision threshold ${pct(result.decision_threshold,0)} · possible referral band ${pct(result.decision_threshold-result.band_halfwidth,0)}–${pct(result.decision_threshold+result.band_halfwidth,0)}. Referral applies within that band only when the conformal model includes both outcomes.</div><p class="decision-meaning">${escapeHtml(decisionMeaning||"No decision label was returned.")}</p></article>${supportCard}<article class="card result-card"><h3>What influenced this estimate?</h3><p class="section-explainer">These are SHAP contributions from the notebook’s base LightGBM in raw log-odds space. They are not a decomposition of the calibrated probability or causes of default.</p>${factors}<p class="result-disclaimer">${escapeHtml(result.full_explanation)}</p>${history}</article><div class="button-row"><button class="button secondary" id="printAssessment">Print / save report</button></div>`;
  $("#printAssessment").addEventListener("click",()=>window.print());
  if (result.cohort_comparison && !result.is_provisional) {
    const refs = result.cohort_comparison;
    const rows = refs.field_comparison.map(row => "<tr><td>" + escapeHtml(labels[row.field] || row.field) +
      "</td><td>" + formatApplicantValue(row.field, row.applicant) + "</td><td>" +
      formatApplicantValue(row.field, row.median) + "</td><td>" +
      formatApplicantValue(row.field, row.q25) + "–" + formatApplicantValue(row.field, row.q75) + "</td></tr>").join("");
    const comparison = document.createElement("article");
    comparison.className = "card result-card";
    comparison.innerHTML = "<h3>Compared with notebook cal_B observations</h3><div class='cohort-percentile'><b>" +
      pct(refs.risk_percentile, 1) + "</b><span>of historical model scores are at or below this applicant’s risk · n=" +
      fmt(refs.risk_cohort_count, 0) + "</span></div><div class='table-scroll'><table class='data-table'><thead><tr>" +
      "<th>Input</th><th>This application</th><th>Typical midpoint</th><th>Middle half of records</th></tr></thead><tbody>" +
      rows + "</tbody></table></div><p class='section-explainer'>The percentile is the share of cohort scores at or below this score. The midpoint is the cohort median; the middle-half range runs from its 25th to 75th percentile. These are context, not targets.</p><p class='result-disclaimer'>" + escapeHtml(refs.dataset) + "</p>";
    $("#assessmentResult").insertBefore(comparison, $("#printAssessment").parentElement);
  }
  if($("#saveHistory").checked) saveLocalAssessment(applicant,result);
}
function formatApplicantValue(field, value) {
  if (["loan_amnt", "annual_inc", "revol_bal"].includes(field)) return money(value);
  if (["dti", "revol_util"].includes(field)) return pct(Number(value) / 100, 1);
  if (["emp_length", "credit_history_years"].includes(field)) return fmt(value, 1) + " yrs";
  return fmt(value, 1);
}
async function submitAssessment(event) {
  event.preventDefault();let loading=false;const submit=$("#runAssessment");if(submit.disabled)return;submit.disabled=true;const error=$("#formError");error.classList.add("hidden");
  try { const applicant=readForm("app");busy(true);loading=true;const result=await api("/api/decide",{method:"POST",body:JSON.stringify(applicant)});const idx=$("#assessmentForm").dataset.example;const ex=idx!=null?state.examples[Number(idx)]:null;const historic=ex&&Object.keys(applicant).every(k=>String(applicant[k])===String(ex.applicant[k]))?ex:null;renderAssessmentResult(result,applicant,historic); }
  catch(e){error.replaceChildren(document.createTextNode(e.message+" "));if(transientError(e)){const retry=document.createElement("button");retry.type="button";retry.className="text-button retry-button";retry.textContent="Retry";retry.addEventListener("click",()=>submitAssessment(new Event("submit",{cancelable:true})),{once:true});error.append(retry);}error.classList.remove("hidden");}finally{if(loading)busy(false);submit.disabled=false;}
}
async function loadPortfolio(filters={}) {
  try {
    const q=new URLSearchParams(Object.entries(filters).filter(([,v])=>v!==""&&v!=null));state.portfolio=await api(`/api/portfolio/summary?${q}`);const d=state.portfolio;
    $("#portfolioKpis").innerHTML=[{label:"Loans in selected cohort",value:fmt(d.count,0),note:d.dataset,meaning:"Records remaining after the filters above.",icon:"▤"},{label:"Observed default rate",value:pct(d.observed_default_rate,2),note:"Actual outcomes",meaning:"Historical share recorded as Charged Off; not a prediction.",icon:"⌁"},{label:"Mean predicted risk",value:pct(d.mean_predicted_risk,2),note:"Frozen calibrated model",meaning:"Average estimated default chance for the selected records.",icon:"◉"},{label:"Mean loan amount",value:money(d.mean_loan_amount),note:"Funded principal in cohort",meaning:"Average original loan amount, not the current unpaid balance.",icon:"$"}].map(renderKpi).join("");
    renderBars($("#purposeBars"),d.by_purpose,"count","name");renderDecisions($("#portfolioDecisions"),d.decisions);
    renderBars($("#loanAmountBars"),d.loan_amount_distribution,"count","range");renderBars($("#homeBars"),d.by_home,"count","name");
    const fill=(id,items,label)=>{const el=$(id),old=el.value;el.innerHTML=`<option value="">All ${label}</option>`+items.map(v=>`<option value="${escapeHtml(v)}">${escapeHtml(String(v).replaceAll("_"," "))}</option>`).join("");if(items.includes(old))el.value=old;};
    fill("#filterPurpose",d.purposes,"purposes");fill("#filterHome",d.home_ownerships,"home types");fill("#filterYear",d.years,"years");
    $("#frozenPolicyText").textContent=`${pct(d.frozen_policy.approve_below,0)} / ${pct(d.frozen_policy.reject_above,0)}`;
    $("#approveSlider").value=d.frozen_policy.approve_below;$("#rejectSlider").value=d.frozen_policy.reject_above;updateSliderLabels();
    $("#approveSlider").min=state.meta.policy_sweep.start;$("#approveSlider").max=state.meta.policy_sweep.end;$("#approveSlider").step=state.meta.policy_sweep.step;
    $("#rejectSlider").min=0;$("#rejectSlider").max=1;$("#rejectSlider").step=state.meta.policy_sweep.step;
    $("#portfolioScope").textContent=`${d.dataset} · ${fmt(d.count,0)} records`;
  } catch(e){$("#portfolioScope").textContent="Portfolio data unavailable.";notice(e.message,"error",()=>loadPortfolio(filters));}
}
function updateSliderLabels(){$("#approveValue").value=pct(Number($("#approveSlider").value),0);$("#rejectValue").value=pct(Number($("#rejectSlider").value),0);}
async function loadGovernance() {
  try {const d=await api("/api/governance");state.governance=d;const entries=Object.entries(d.models);
    $("#governanceScope").textContent=`${d.evaluation_dataset} · Calibration: ${d.calibration}`;
    const active=d.models.lightgbm_calibrated;
    $("#governanceKpis").innerHTML=[{label:"ROC-AUC",value:fmt(active.roc_auc,4),note:"2015 OOT · read-only",meaning:"Ranks defaulting loans above paid loans; closer to 1 indicates stronger separation.",icon:"⌁"},{label:"Brier score",value:fmt(active.brier_score,4),note:"Calibrated 2015 OOT probabilities",meaning:"Average squared error in probability estimates; closer to 0 is better.",icon:"◉"},{label:"Expected calibration error",value:fmt(active.ece,4),note:"10 equal-width bins",meaning:"Average gap between estimated chances and observed default shares across risk bands.",icon:"◎"},{label:"Model features",value:fmt(d.features.length,0),note:"Notebook preprocessor output columns",meaning:"Encoded and scaled columns presented to the saved estimator.",icon:"✳"},{label:"Selected threshold",value:fmt(d.threshold_selection.selected_threshold,2),note:`8% ROI hurdle · matches saved: ${d.threshold_selection.matches_saved_notebook?"yes":"no"}`,meaning:"Chosen by the notebook’s 2014 profit sweep after subtracting an 8% return hurdle from principal. The 2015 set played no role in choosing it.",icon:"⌁"},{label:"Training APR reference",value:pct(d.apr_reference.value,2),note:"Training mean int_rate / 100",meaning:"Historical average interest rate in training data; shown for context, not an applicant price.",icon:"◈"}].map(renderKpi).join("");
    const metricLabels={roc_auc:"ROC-AUC",pr_auc:"PR-AUC",brier_score:"Brier score",brier_skill_score:"Brier skill",ece:"Calibration error (ECE)",log_loss:"Log loss",ks:"KS separation"};
    const metrics=["roc_auc","pr_auc","brier_score","brier_skill_score","ece","log_loss","ks"];
    $("#modelTable").innerHTML=`<thead><tr><th>Saved model</th>${metrics.map(m=>`<th title="${escapeHtml(metricLabels[m])}">${metricLabels[m]}</th>`).join("")}</tr></thead><tbody>${entries.map(([key,row])=>`<tr><td>${escapeHtml(row.model_label||key)}</td>${metrics.map(m=>`<td>${fmt(row[m],4)}</td>`).join("")}</tr>`).join("")}</tbody>`;
    drawXYChart($("#rocChart"),d.roc_curve,"x","y",true,"Final out-of-time ROC curve","False positive rate","True positive rate");drawXYChart($("#prChart"),d.precision_recall_curve,"x","y",false,"Final out-of-time precision recall curve","Recall","Precision");drawXYChart($("#calibrationChart"),d.calibration_curve,"predicted","observed",true,"Final out-of-time calibration curve","Predicted probability","Observed default rate");
    renderBars($("#featureBars"),d.feature_importance,"importance","feature");
    const splitNames={train:"Training data",early_stop:"Early-stop data",calibration:"2014 calibration data",test_oot:"Final test data (2015)"};const splitMeaning={train:"Used to fit model parameters",early_stop:"Used to choose when training stops",calibration:"Used for calibration and policy selection",test_oot:"Held aside for final evaluation"};
    $("#splitCards").innerHTML=Object.entries(d.split_counts).map(([k,count])=>`<div class="split-row"><strong>${escapeHtml(splitNames[k]||k)}<small>${escapeHtml(splitMeaning[k]||"")}</small></strong><span>${fmt(count,0)} loans</span><b>${pct(d.split_default_rates[k],2)}<small>observed defaults</small></b></div>`).join("");
    const hashes=Object.entries(d.integrity);const matches=hashes.filter(([,v])=>v.matches).length;$("#integrityStatus").innerHTML=`<i></i> ${matches===hashes.length?'All checks passed':'Check mismatch'}`;$("#integrityStatus").classList.toggle("bad",matches!==hashes.length);
    $("#hashList").innerHTML=hashes.map(([name,item])=>`<div class="hash-row"><span class="hash-name" title="${escapeHtml(item.actual||'')}">${escapeHtml(name)}</span><span class="hash-state ${item.matches?'':'bad'}">${item.matches?'MATCH':'CHECK'}</span></div>`).join("");
  }catch(e){$("#governanceScope").textContent="Governance data could not be loaded.";notice(e.message,"error",loadGovernance);}
}
const historyStorageKey="creditview.assessmentHistory",legacyHistoryStorageKey="finclarity.assessmentHistory";
function getAssessmentHistory(){const current=localStorage.getItem(historyStorageKey);if(current)return current;const legacy=localStorage.getItem(legacyHistoryStorageKey);if(legacy){localStorage.setItem(historyStorageKey,legacy);return legacy;}return null;}
function saveLocalAssessment(applicant,result){try{const rows=JSON.parse(getAssessmentHistory()||"[]");rows.push({saved_at:new Date().toISOString(),applicant,decision:result.decision,formal_status:result.formal_status||result.decision,indicative_recommendation:result.indicative_recommendation||result.decision,indicative_reason:result.indicative_reason||null,p_default:result.p_default,is_provisional:Boolean(result.is_provisional),support_warnings:result.support_warnings||[],formal_explanation:result.formal_explanation||result.full_explanation,model_version:result.model_version});localStorage.setItem(historyStorageKey,JSON.stringify(rows));notice("Assessment saved in this browser.");}catch{notice("Browser storage is unavailable; the assessment was not saved.","error");}}
function exportHistory(){try{const text=getAssessmentHistory();if(!text){notice("No local assessment history to export.");return;}download("creditview-history.json",text,"application/json");}catch{notice("Could not read local history.","error");}}
async function runWhatIf(){let loading=false;const err=$("#scenarioError");err.textContent="";try{const original=readForm("base",state.meta.fields.map(f=>f.name)),modified=readForm("changed",state.meta.fields.map(f=>f.name));busy(true);loading=true;const r=await api("/api/whatif",{method:"POST",body:JSON.stringify({original_application:original,modified_application:modified})});state.scenarioApplicant=original;
  const supportNotes=[...[['Starting application',r.original],['Changed application',r.modified]].filter(([,v])=>v.is_provisional).map(([name,v])=>`${name}: ${v.support_warnings.join(' ')}`)];
  const originalFormal=r.original.formal_status||r.original.decision,modifiedFormal=r.modified.formal_status||r.modified.decision;
  $("#whatIfResult").innerHTML=`<article class="card whatif-card"><div class="whatif-metric"><span>Starting application · estimated default chance</span><b>${pct(r.original.p_default,2)}</b><small>${r.original.is_provisional?'Provisional estimate; may be unreliable.':'Model estimate for the first set of inputs.'}</small></div><div class="whatif-metric"><span>Changed application · estimated default chance</span><b>${pct(r.modified.p_default,2)}</b><small>${r.modified.is_provisional?'Provisional estimate; may be unreliable.':'Re-scored after applying your edited values.'}</small></div><div class="whatif-metric"><span>Change in estimate</span><b>${r.delta_p_default>0?'+':''}${(r.delta_p_default*100).toFixed(2)} percentage points</b><small>Scenario estimate minus starting estimate; positive means higher estimated risk.</small></div><div class="whatif-metric"><span>Formal assessment status</span><b>${escapeHtml(originalFormal)} → ${escapeHtml(modifiedFormal)}</b><small>Unsupported applications remain subject to manual review.</small></div><div class="whatif-metric"><span>Indicative model recommendation</span><b>${r.original.is_provisional?'INDICATIVE ':''}${escapeHtml(r.original.indicative_recommendation)} → ${r.modified.is_provisional?'INDICATIVE ':''}${escapeHtml(r.modified.indicative_recommendation)}</b><small>Uses the saved threshold and conformal uncertainty rule. An indicative outcome is not a lending decision.</small></div><div class="whatif-detail"><b>Inputs you changed:</b> ${r.key_drivers.map(escapeHtml).join(" · ")||"None"}<br>${supportNotes.length?`<b>Support warning:</b> ${supportNotes.map(escapeHtml).join(' · ')}<br>`:''}The estimates use the same notebook artifacts. This comparison does not show that changing an input causes a real-world outcome.</div></article>`;
  }catch(e){err.textContent=e.message;if(transientError(e))addRetry(err,runWhatIf);}finally{if(loading)busy(false);}}
async function runPolicy(){const ap=Number($("#approveSlider").value),rj=Number($("#rejectSlider").value);const el=$("#policyResult");if(ap>=rj){el.innerHTML="<span class='field-error'>The lower boundary must be smaller than the upper boundary.</span>";return;}el.textContent="Applying these boundaries to notebook observations…";try{const d=await api("/api/policy/evaluate",{method:"POST",body:JSON.stringify({approve_below:ap,reject_above:rj})});el.innerHTML=`<div class="policy-metric-grid">${[["Marked Approve",pct(d.approval_rate,2),"Share of selected loans below the lower boundary and not referred."],["Sent for review",pct(d.referral_rate,2),"Share referred by support checks or the conformal uncertainty rule."],["Marked Reject",pct(d.rejection_rate,2),"Share assigned to Reject under these boundaries."],["Later default rate among Approve",pct(d.approved_default_rate,2),"Observed Charged Off share within the group marked Approve."]].map(([a,b,c])=>`<div class="policy-metric"><span>${a}</span><b>${b}</b><small>${c}</small></div>`).join("")}</div><p class="muted small">${fmt(d.count,0)} actual 2014 cal_B records · center threshold ${pct(d.thresholds.notebook_threshold,2)} · referral half-width ±${pct(d.thresholds.band_halfwidth,2)}. These are your exploratory boundaries; the model’s saved rule remains unchanged.</p>`;}catch(e){el.replaceChildren(document.createTextNode(e.message));if(transientError(e))addRetry(el,runPolicy);}}
function parseCsv(text){const lines=text.trim().split(/\r?\n/);if(lines.length<2)throw new Error("CSV needs a header and at least one applicant row.");const parseLine=line=>{const out=[];let cur="",quoted=false;for(let i=0;i<line.length;i++){const c=line[i];if(c==='"'&&line[i+1]==='"'&&quoted){cur+='"';i++;}else if(c==='"')quoted=!quoted;else if(c===','&&!quoted){out.push(cur.trim());cur="";}else cur+=c;}out.push(cur.trim());return out;};const headers=parseLine(lines.shift()).map(x=>x.replace(/^\uFEFF/,""));return lines.filter(x=>x.trim()).map((line,index)=>{const values=parseLine(line);const obj={};headers.forEach((h,i)=>obj[h]=values[i]??"");return {row:index+2,obj};});}
async function scoreCsv(){
  let loading=false;const file=$("#batchFile").files[0],el=$("#batchResult");
  if(!file){el.textContent="Choose a CSV file first.";return;}
  const maxBytes=Number(state.meta.app_limits.max_upload_bytes);
  if(file.size>maxBytes){el.textContent=`File exceeds the local ${fmt(maxBytes/1048576,0)} MB limit.`;return;}
  try{
    const parsed=parseCsv(await file.text());const names=state.meta.fields.map(f=>f.name);const errors=[];
    const apps=parsed.map(({row,obj})=>{
      const missing=names.filter(k=>obj[k]===undefined||obj[k]==="");
      if(missing.length)errors.push(`Row ${row}: missing ${missing.join(", ")}`);
      const out={};
      for(const k of names){
        const f=state.meta.fields.find(x=>x.name===k);const value=f.type==="select"?obj[k]:Number(obj[k]);out[k]=value;
        if(f.type==="select"?!f.options?.includes(value):(!Number.isFinite(value)||(f.min!=null&&(f.exclusive_min?value<=f.min:value<f.min))||(f.max!=null&&(f.exclusive_max?value>=f.max:value>f.max))))errors.push("Row "+row+": invalid "+(labels[k]||k));
      }
      return out;
    });
    if(errors.length)throw new Error(errors.slice(0,4).join("; "));
    if(apps.length>Number(state.meta.app_limits.max_batch_rows))throw new Error(`Maximum batch size is ${fmt(state.meta.app_limits.max_batch_rows,0)} rows.`);
    busy(true);loading=true;
    const result=await api("/api/batch",{method:"POST",body:JSON.stringify(apps)});
    const csvCell=value=>`"${String(value??"").replaceAll('"','""')}"`;
    const csvRows=result.results.map(r=>[
      r.row,r.formal_status||r.decision,r.is_provisional?`Indicative ${r.indicative_recommendation}`:r.indicative_recommendation,
      r.p_default,r.is_provisional?"Provisional; may be unreliable":"Model estimate",r.is_provisional,
      (r.support_warnings||[]).join("; "),r.formal_explanation,
    ].map(csvCell).join(","));
    const rows=["row_number,formal_status,indicative_recommendation,estimated_default_probability,risk_estimate_status,provisional_prediction,support_warnings,formal_explanation",...csvRows].join("\n");
    const reviewCount=result.results.filter(r=>r.is_provisional).length;
    const tableRows=result.results.map(r=>{
      const formal=r.formal_status||r.decision;
      const indicative=r.is_provisional?`Indicative ${r.indicative_recommendation}`:r.indicative_recommendation;
      const warnings=(r.support_warnings||[]).map(escapeHtml).join("<br>")||"—";
      return `<tr><td>${fmt(r.row,0)}</td><td>${escapeHtml(formal)}</td><td>${escapeHtml(indicative)}</td><td>${pct(r.p_default,2)}${r.is_provisional?"<br><small>Provisional; may be unreliable</small>":""}</td><td>${warnings}</td><td>${escapeHtml(r.formal_explanation||"")}</td></tr>`;
    }).join("");
    el.innerHTML=`<p>Scored ${fmt(result.count,0)} rows; ${fmt(reviewCount,0)} require manual review. Model risk and formal status remain separate.</p><div class="table-scroll"><table class="data-table"><thead><tr><th>Row</th><th>Formal assessment</th><th>Indicative model recommendation</th><th>Estimated default risk</th><th>Training-support warnings</th><th>Status explanation</th></tr></thead><tbody>${tableRows}</tbody></table></div><button class="button secondary" id="downloadBatch">Download results CSV</button>`;
    $("#downloadBatch").addEventListener("click",()=>download("creditview-scores.csv",rows,"text/csv"));
  }catch(e){el.textContent=e.message;if(transientError(e))addRetry(el,scoreCsv);}
  finally{if(loading)busy(false);}
}
function bind(){
  $$(".nav-link").forEach(b=>b.addEventListener("click",()=>showPage(b.dataset.page)));$$("[data-go]").forEach(b=>b.addEventListener("click",()=>showPage(b.dataset.go)));
  $("#newAssessment").addEventListener("click",()=>{showPage("assessment");$("#assessmentForm").reset();$("#assessmentResult").innerHTML="";});$("#menuToggle").addEventListener("click",()=>$(".nav").classList.toggle("open"));
  $("#themeToggle").addEventListener("click",()=>{const dark=document.body.classList.toggle("dark");try{localStorage.setItem("creditview.theme",dark?"dark":"light");}catch{}$("#themeToggle").setAttribute("aria-pressed",String(dark));document.querySelector("meta[name=theme-color]").content=dark?"#151d19":"#f5f3ed";});
  try{if((localStorage.getItem("creditview.theme")||localStorage.getItem("finclarity.theme"))==="dark"){document.body.classList.add("dark");$("#themeToggle").setAttribute("aria-pressed","true");document.querySelector("meta[name=theme-color]").content="#151d19";}}catch{}
  $("#assessmentForm").addEventListener("submit",submitAssessment);$("#resetAssessment").addEventListener("click",()=>{setTimeout(()=>{$("#assessmentResult").innerHTML=`<article class="card empty-state"><h2>Your assessment appears here</h2><p>Enter all required applicant inputs to run the saved model.</p></article>`;$("#formError").classList.add("hidden");},0);});
  $("#loadExample").addEventListener("click",()=>{try{const ex=applicantFromExample($("#examplePicker").value);putForm("app",ex.applicant);notice(`Loaded a real notebook cal_B record from ${ex.originated}; historical outcome: ${ex.historical_outcome}.`);$("#assessmentForm").dataset.example=String($("#examplePicker").value);}catch(e){notice(e.message,"error");}});
  $("#loadScenarioExample").addEventListener("click",()=>{try{const ex=applicantFromExample($("#scenarioExample").value);putForm("base",ex.applicant);putForm("changed",ex.applicant);notice(`Loaded notebook cal_B record from ${ex.originated}.`);}catch(e){notice(e.message,"error");}});
  $("#copyBaseline").addEventListener("click",()=>{try{putForm("changed",readForm("base",state.meta.fields.map(f=>f.name)));notice("Starting application values copied to the scenario.");}catch(e){notice(e.message,"error");}});
  lockAction($("#runWhatIf"),runWhatIf);
  lockAction($("#applyFilters"),()=>{const min=$("#filterFicoMin").value,max=$("#filterFicoMax").value;if(min&&max&&Number(min)>Number(max)){notice("FICO minimum must not exceed the maximum.","error");return;}return loadPortfolio({purpose:$("#filterPurpose").value,home_ownership:$("#filterHome").value,year:$("#filterYear").value,fico_min:min,fico_max:max});});
  $("#approveSlider").addEventListener("input",updateSliderLabels);$("#rejectSlider").addEventListener("input",updateSliderLabels);lockAction($("#runPolicy"),runPolicy);lockAction($("#scoreCsv"),scoreCsv);
  $("#downloadBatchTemplate").addEventListener("click",downloadBatchTemplate);
  $("#exportHistory").addEventListener("click",exportHistory);$("#clearHistory").addEventListener("click",()=>{if(confirm("Delete all CreditView assessment history saved in this browser?")){localStorage.removeItem(historyStorageKey);localStorage.removeItem(legacyHistoryStorageKey);notice("Local history cleared.");}});
  lockAction($("#refreshGovernance"),()=>{state.governance=null;return loadGovernance();});
  window.addEventListener("hashchange",()=>{const p=location.hash.slice(1);if(["dashboard","assessment","scenario","portfolio","governance"].includes(p))showPage(p);});
}
async function init(){bind();try{state.meta=await api("/api/meta");renderBatchFieldGuide();renderFieldForm("#assessmentFields","app");renderFieldForm("#baselineFields","base",true);renderFieldForm("#changedFields","changed",true);await loadExamples();const page=location.hash.slice(1);showPage(["dashboard","assessment","scenario","portfolio","governance"].includes(page)?page:"dashboard");await loadDashboard();}catch(e){notice(e.message,"error",()=>location.reload());}}
function download(name,content,type){const a=document.createElement("a");a.href=URL.createObjectURL(new Blob([content],{type}));a.download=name;document.body.append(a);a.click();a.remove();window.setTimeout(()=>URL.revokeObjectURL(a.href),1000);}
document.addEventListener("DOMContentLoaded",init);
