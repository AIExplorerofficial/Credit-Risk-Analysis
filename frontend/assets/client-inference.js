"use strict";

const OUTSIDE_SUPPORT = "Manual Review Required — Outside Model Support";

function formFeatures(applicant, model) {
  const values = new Array(model.feature_columns.length).fill(0);
  const index = new Map(model.feature_columns.map((name, i) => [name, i]));
  const engineered = {
    ...applicant,
    fico_avg: Number(applicant.fico_avg),
    emp_length: Number(applicant.emp_length ?? 0),
    pub_rec: Number(applicant.pub_rec ?? 0),
    delinq_2yrs: Number(applicant.delinq_2yrs ?? 0),
    inq_last_6mths: Number(applicant.inq_last_6mths ?? 0),
  };

  for (const [source, transform] of Object.entries(model.preprocessing.numeric)) {
    const raw = Number(engineered[source]);
    const imputed = Number.isFinite(raw) ? raw : transform.impute;
    values[index.get(transform.feature)] = (imputed - transform.mean) / transform.scale;
  }

  for (const [source, spec] of Object.entries(model.preprocessing.categorical)) {
    let category = String(engineered[source] ?? "OTHER");
    if (source === "home_ownership" && !["RENT", "OWN", "MORTGAGE"].includes(category)) {
      category = "OTHER";
    }
    const featureCategory = spec.categories.includes(category)
      ? (spec.infrequent.includes(category) ? "infrequent_sklearn" : category)
      : (spec.infrequent.length ? "infrequent_sklearn" : null);
    const featureName = featureCategory == null ? null : spec.encoded_features[featureCategory];
    if (featureName != null) values[index.get(featureName)] = 1;
  }
  return values;
}

function treeValue(node, features) {
  if (Object.hasOwn(node, "leaf_value")) return node.leaf_value;
  const value = features[node.split_feature];
  const goesLeft = Number.isNaN(value) ? node.default_left : value <= node.threshold;
  return treeValue(goesLeft ? node.left_child : node.right_child, features);
}

function rawMargin(features, model) {
  let margin = Number(model.tree_model.average_output ? 1 : 0);
  for (const tree of model.tree_model.tree_info) {
    margin += treeValue(tree.tree_structure, features);
  }
  return margin;
}

function sigmoidProbability(features, model) {
  const margin = rawMargin(features, model);
  const z = -(model.calibration.a * margin + model.calibration.b);
  return z >= 0 ? 1 / (1 + Math.exp(-z)) : Math.exp(z) / (1 + Math.exp(z));
}

function baseProbability(features, model) {
  const margin = rawMargin(features, model);
  return margin >= 0 ? 1 / (1 + Math.exp(-margin)) : Math.exp(margin) / (1 + Math.exp(margin));
}

function decideProbability(pDefault, predictionSet, outOfSupport, model) {
  const both = predictionSet[0] && predictionSet[1];
  const ambiguous = both && Math.abs(pDefault - model.decision.threshold) <= model.decision.band_halfwidth;
  const indicative = ambiguous ? "Refer" : (pDefault < model.decision.threshold ? "Approve" : "Reject");
  return {
    decision: outOfSupport ? "Refer" : indicative,
    reason: outOfSupport ? "out-of-support" : (ambiguous ? "ambiguous" : ""),
  };
}

function scoreTransformed(features, model) {
  const margin = rawMargin(features, model);
  const z = -(model.calibration.a * margin + model.calibration.b);
  const pDefault = z >= 0 ? 1 / (1 + Math.exp(-z)) : Math.exp(z) / (1 + Math.exp(z));
  const baseProbability = margin >= 0 ? 1 / (1 + Math.exp(-margin)) : Math.exp(margin) / (1 + Math.exp(margin));
  const q = model.conformal.quantile;
  const predictionSet = [pDefault <= q, (1 - pDefault) <= q];
  return { margin, pDefault, baseProbability, predictionSet };
}

function prepareShapTrees(model) {
  if (model._shapTrees) return model._shapTrees;
  const prepared = [];
  let maxDepth = 0;
  for (const tree of model.tree_model.tree_info) {
    // SHAP's LightGBM adapter indexes split nodes by split_index and leaves
    // after them by leaf_index. Preserve that layout so parent pointers and
    // path state follow the same topology as the notebook TreeExplainer.
    const internalCount = Number(tree.num_leaves) - 1;
    const nodes = new Array(Number(tree.num_leaves) * 2 - 1);
    // SHAP's C++ Node builder uses node 0 as the root's parent sentinel. The
    // traversal relies on this to stop when it unwinds to the root.
    function visit(source, parent = 0, parentFeature = -1, depth = 0) {
      const leaf = Object.hasOwn(source, "leaf_value");
      const id = leaf ? internalCount + Number(source.leaf_index) : Number(source.split_index);
      const entry = { left: -1, right: -1, default: -1, parent, parentFeature,
        feature: leaf ? -1 : source.split_feature,
        threshold: leaf ? 0 : Math.fround(source.threshold),
        value: Math.fround(leaf ? source.leaf_value : source.internal_value),
        from: 0 };
      nodes[id] = entry;
      if (!leaf) {
        maxDepth = Math.max(maxDepth, depth + 1);
        entry.left = visit(source.left_child, id, entry.feature, depth + 1);
        entry.right = visit(source.right_child, id, entry.feature, depth + 1);
        entry.default = source.default_left ? entry.left : entry.right;
      }
      return id;
    }
    visit(tree.tree_structure);
    prepared.push(nodes);
  }
  const depth = Math.max(1, maxDepth);
  const weights = new Float32Array((depth + 1) * (depth + 1));
  function binomial(n, k) {
    let result = 1;
    if (k > n - k) k = n - k;
    for (let i = 0; i < k; i++) { result *= n - i; result /= i + 1; }
    return result;
  }
  for (let n = 0; n <= depth; n++) {
    for (let m = 0; m <= depth; m++) {
      weights[n + depth * m] = Math.fround(1 / (n * binomial(n - 1, m)));
    }
  }
  model._shapTrees = { trees: prepared, depth, weights };
  return model._shapTrees;
}

function treeShapIndependent(target, reference, nodes, contributions, depth, weights) {
  const featureCount = target.length;
  const history = new Int16Array(featureCount);
  const positive = new Float32Array(nodes.length);
  const negative = new Float32Array(nodes.length);
  const stack = new Int32Array(depth + 2);
  let stackCount = 0, nodeId = 0, nextNode = -1, fromChild = -1, n = 0, m = 0;
  const root = nodes[0];
  if (root.left < 0) { contributions[featureCount] += root.value; return; }

  function branch(features, feature, node) {
    if (Number.isNaN(features[feature])) return node.default;
    return features[feature] > node.threshold ? node.right : node.left;
  }
  let current = root;
  let nextX = branch(target, current.feature, current);
  let nextR = branch(reference, current.feature, current);
  if (nextX !== nextR) { nodes[nextX].from = 1; nodes[nextR].from = 2; }
  else nodes[nextX].from = 0;
  if (nextX === nextR) nextNode = nextX;
  if (nextNode < 0) {
    nextNode = current.left;
    if (nextR === nextNode) { n++; history[current.feature]--; }
    else if (nextX === nextNode) { m++; n++; history[current.feature]++; }
  }
  stack[stackCount++] = nodeId;

  while (true) {
    nodeId = nextNode;
    current = nodes[nodeId];
    const feature = current.feature;
    const parentFeature = current.parentFeature;
    const fromFlag = current.from;
    if (current.left < 0) {
      if (m === 0) contributions[featureCount] += current.value;
      if (n !== 0) {
        if (m !== 0) positive[nodeId] = Math.fround(current.value * weights[n + depth * (m - 1)]);
        if (m !== n) negative[nodeId] = Math.fround(-current.value * weights[n + depth * m]);
      }
      stackCount--;
      nextNode = stack[stackCount];
      fromChild = nodeId;
      if (history[parentFeature] > 0) history[parentFeature]--;
      else if (history[parentFeature] < 0) history[parentFeature]++;
      if (history[parentFeature] === 0) {
        if (fromFlag === 1) { n--; m--; }
        else if (fromFlag === 2) n--;
      }
      continue;
    }

    nextX = branch(target, feature, current);
    nextR = branch(reference, feature, current);
    if (nextX >= 0) {
      if (nextX !== nextR) { nodes[nextX].from = 1; nodes[nextR].from = 2; }
      else nodes[nextX].from = 0;
    }

    if (fromChild === -1) {
      stack[stackCount++] = nodeId;
      nextNode = -1;
      if (history[feature] > 0) { nextNode = nextX; history[feature]++; }
      else if (history[feature] < 0) { nextNode = nextR; history[feature]--; }
      if (nextNode < 0 && nextX === nextR) nextNode = nextX;
      if (nextNode >= 0) continue;
      nextNode = current.left;
      if (nextR === nextNode) { n++; history[feature]--; }
      else if (nextX === nextNode) { m++; n++; history[feature]++; }
      fromChild = -1;
      continue;
    }

    nextNode = (nextR === nextX || history[feature] !== 0) ? current.parent : -1;
    if (nextNode >= 0) {
      if (nodeId === 0) break;
      positive[nodeId] = positive[fromChild];
      negative[nodeId] = negative[fromChild];
      fromChild = nodeId;
      stackCount--;
      if (history[parentFeature] > 0) history[parentFeature]--;
      else if (history[parentFeature] < 0) history[parentFeature]++;
      if (history[parentFeature] === 0) {
        if (fromFlag === 1) { n--; m--; }
        else if (fromFlag === 2) n--;
      }
      continue;
    }
    if (fromChild === current.left) {
      stack[stackCount++] = nodeId;
      nextNode = current.right;
      if (nextX === nextNode) { m++; n++; history[feature]++; }
      else if (nextR === nextNode) { n++; history[feature]--; }
      fromChild = -1;
      continue;
    }
    if (fromChild === current.right) {
      let posX=0,negX=0,posR=0,negR=0;
      if (nextX===current.right&&nextR===current.left) {
        posX=positive[current.right];negX=negative[current.right];posR=positive[current.left];negR=negative[current.left];
      } else if (nextX===current.left&&nextR===current.right) {
        posX=positive[current.left];negX=negative[current.left];posR=positive[current.right];negR=negative[current.right];
      }
      contributions[feature] += posX + negR;
      positive[nodeId]=Math.fround(posX+posR);
      negative[nodeId]=Math.fround(negX+negR);
      if (nodeId===0) break;
      stackCount--;
      nextNode=stack[stackCount];
      fromChild=nodeId;
      if (history[parentFeature]>0)history[parentFeature]--;
      else if(history[parentFeature]<0)history[parentFeature]++;
      if(history[parentFeature]===0){if(fromFlag===1){n--;m--;}else if(fromFlag===2)n--;}
    }
  }
}

function interventionalShap(features, model) {
  const { trees, depth, weights } = prepareShapTrees(model);
  const background = model.shap.background;
  if (!Array.isArray(background) || background.length !== model.shap.background_rows) {
    throw new Error("The notebook SHAP background is missing or incomplete.");
  }
  const total = new Float64Array(features.length + 1);
  for (const reference of background) {
    const perReference = new Float64Array(features.length + 1);
    for (const tree of trees) treeShapIndependent(features, reference, tree, perReference, depth, weights);
    for (let i = 0; i < total.length; i++) total[i] += perReference[i] / background.length;
  }
  return Array.from(total.subarray(0, features.length));
}

const REASON_TEMPLATES = {
  fico_avg:{increases_risk:"Credit score (FICO {val}) is lower than optimal, increasing estimated risk of default.",decreases_risk:"Strong credit score (FICO {val}) provides substantial positive creditworthiness support."},
  dti:{increases_risk:"Debt-to-income ratio ({val}%) indicates high monthly debt burden relative to income.",decreases_risk:"Low debt-to-income ratio ({val}%) indicates manageable debt obligations."},
  revol_util:{increases_risk:"Revolving line utilization ({val}%) is elevated, indicating heavy revolving credit reliance.",decreases_risk:"Disciplined revolving utilization ({val}%) indicates credit line buffer."},
  annual_inc:{increases_risk:"Annual income (${val}) is on the lower side for the requested borrowing profile.",decreases_risk:"Annual income (${val}) demonstrates strong capacity to service debt."},
  loan_amnt:{increases_risk:"Requested loan amount (${val}) represents substantial principal exposure.",decreases_risk:"Moderate loan amount (${val}) reduces principal risk."},
  delinq_2yrs:{increases_risk:"History of delinquency ({val} in past 2 years) signals elevated credit risk.",decreases_risk:"Clean 2-year delinquency history supports reliable repayment tendency."},
  inq_last_6mths:{increases_risk:"Recent credit inquiries ({val} in past 6 months) indicates active credit seeking.",decreases_risk:"Few recent credit inquiries indicates stable credit demand."},
  credit_history_years:{increases_risk:"Shorter credit history ({val} years) limits historical repayment track record.",decreases_risk:"Established credit history ({val} years) provides solid repayment evidence."},
  revol_bal:{increases_risk:"Revolving balance (${val}) represents substantial outstanding debt.",decreases_risk:"Low revolving balance (${val}) reflects limited outstanding debt burden."},
  open_acc:{increases_risk:"Number of open accounts ({val}) suggests broader exposure across credit lines.",decreases_risk:"Active open accounts ({val}) demonstrates active credit management."},
  total_acc:{increases_risk:"High total account count ({val}) indicates extensive credit activity history.",decreases_risk:"Established credit account history ({val} total accounts) confirms credit familiarity."},
  emp_length:{increases_risk:"Shorter employment tenure ({val} years) may affect income stability.",decreases_risk:"Stable employment history ({val} years) reinforces income reliability."},
  pub_rec:{increases_risk:"Presence of derogatory public records ({val}) elevates risk assessment.",decreases_risk:"Clean public record history supports overall borrower integrity."},
  home_ownership:{increases_risk:"Housing ownership status ({val}) carries higher risk profile in historical data.",decreases_risk:"Housing stability ({val}) provides supportive borrower stability."},
  purpose:{increases_risk:"Stated loan purpose ({val}) is associated with higher default frequency.",decreases_risk:"Stated loan purpose ({val}) is historically associated with lower default frequency."},
};

function topReasonCodes(shapRow, featureNames, rawValues, topK=3) {
  const grouped=new Map();
  for(let i=0;i<featureNames.length;i++){
    const name=featureNames[i];
    const source=name.startsWith("cat__home_ownership")?"home_ownership":name.startsWith("cat__purpose")?"purpose":name.split("__").at(-1);
    grouped.set(source,(grouped.get(source)||0)+shapRow[i]);
  }
  return [...grouped].sort((a,b)=>Math.abs(b[1])-Math.abs(a[1])).slice(0,topK).map(([feature,value])=>{
    const direction=value>0?"increases_risk":"decreases_risk";
    const raw=Object.hasOwn(rawValues,feature)?rawValues[feature]:null;
    let display=raw;
    if(raw!=null){
      if(["annual_inc","loan_amnt","revol_bal"].includes(feature))display=Number(raw).toLocaleString("en-US",{maximumFractionDigits:0});
      else if(["fico_avg","dti"].includes(feature))display=String(Number(raw));
      else if(["delinq_2yrs","inq_last_6mths","open_acc","total_acc","emp_length","pub_rec"].includes(feature))display=Number(raw).toFixed(0);
      else if(feature==="credit_history_years")display=Number(raw).toFixed(1);
    }
    const template=REASON_TEMPLATES[feature]?.[direction];
    const reasonText=raw==null?`${feature} (${direction.replaceAll("_"," ")}: ${value>=0?"+":""}${value.toFixed(3)} log-odds)`:
      template?template.replace("{val}",String(display)):`${feature} (${direction.replaceAll("_"," ")}: ${value>=0?"+":""}${value.toFixed(3)} log-odds)`;
    return {feature,shap_value:Number(value.toFixed(4)),direction,raw_value:raw,reason_text:reasonText};
  });
}

function supportWarnings(applicant, model) {
  const support = model.support;
  const values = { ...applicant, fico_avg: Number(applicant.fico_avg) };
  const warnings = [];
  for (const [name, limits] of Object.entries(support.support_numeric)) {
    const value = Number(values[name]);
    if (!Number.isFinite(value)) continue;
    const label = name.replaceAll("_", " ");
    if (value < limits.min || value > limits.max) {
      warnings.push(`${label[0].toUpperCase()}${label.slice(1)} ${formatGeneral(value)} is outside the training observations (${formatGeneral(limits.min)}–${formatGeneral(limits.max)}); this is an extrapolation.`);
    } else if (value < limits.typical_min || value > limits.typical_max) {
      warnings.push(`${label[0].toUpperCase()}${label.slice(1)} ${formatGeneral(value)} is unusual compared with the training data (Tukey 1.5×IQR typical range ${formatGeneral(limits.typical_min)}–${formatGeneral(limits.typical_max)}).`);
    }
  }
  for (const [name, categories] of Object.entries(support.support_categories)) {
    const value = String(applicant[name]);
    if (!categories.includes(value)) {
      const label = name.replaceAll("_", " ");
      warnings.push(`${label[0].toUpperCase()}${label.slice(1)} category '${value}' was not observed in the training split.`);
    }
  }
  return warnings;
}

function formatGeneral(value) {
  return Number(value).toLocaleString("en-US", { useGrouping: false, maximumSignificantDigits: 6 });
}

function scoreApplicant(applicant, model, { includeShap = true } = {}) {
  const features = formFeatures(applicant, model);
  const scoredModel = scoreTransformed(features, model);
  const pDefault = scoredModel.pDefault;
  const includesPaid = scoredModel.predictionSet[0];
  const includesDefault = scoredModel.predictionSet[1];
  const warnings = supportWarnings(applicant, model);
  const outside = warnings.some((warning) => warning.includes("is outside the training observations") || warning.includes("was not observed"));
  const decision = decideProbability(pDefault, scoredModel.predictionSet, outside, model);
  const ambiguous = decision.reason === "ambiguous";
  const indicative = ambiguous ? "Refer" : (pDefault < model.decision.threshold ? "Approve" : "Reject");
  const indicativeReason = ambiguous ? "ambiguous" : null;
  const formal = outside ? OUTSIDE_SUPPORT : indicative;
  const formalExplanation = outside
    ? "This application is outside the observed training support. Manual review is required; the model's risk estimate and indicative recommendation may be unreliable for these inputs."
    : indicativeReason === "ambiguous"
      ? `The conformal model considers both outcomes plausible and estimated risk is within ${(model.decision.band_halfwidth * 100).toFixed(1)}% of the ${(model.decision.threshold * 100).toFixed(1)}% threshold, so the application is referred.`
      : indicative === "Approve"
        ? `Estimated risk is below the saved ${(model.decision.threshold * 100).toFixed(1)}% threshold.`
        : `Estimated risk is at or above the saved ${(model.decision.threshold * 100).toFixed(1)}% threshold.`;
  if (outside) warnings.push("This risk estimate is provisional and may be unreliable outside validated training support. A human reviewer must assess the application.");
  const topK = Number(model.shap.top_k_reasons);
  if (includeShap && (!Number.isInteger(topK) || topK < 1)) {
    throw new Error("The notebook's SHAP reason count is missing or invalid.");
  }
  const reasonFactors = includeShap
    ? topReasonCodes(interventionalShap(features, model), model.feature_columns, applicant, topK)
    : [];
  return {
    transformed: features,
    base_probability: scoredModel.baseProbability,
    prediction_set: scoredModel.predictionSet,
    p_default: Number(pDefault.toFixed(4)),
    decision: formal,
    formal_status: formal,
    indicative_recommendation: indicative,
    indicative_reason: indicativeReason,
    refer_reason: outside ? "out-of-support" : indicativeReason,
    decision_threshold: model.decision.threshold,
    band_halfwidth: model.decision.band_halfwidth,
    approve_below: model.decision.threshold - model.decision.band_halfwidth,
    reject_above: model.decision.threshold + model.decision.band_halfwidth,
    is_out_of_support: outside,
    is_provisional: outside,
    support_warnings: warnings,
    top_factors: reasonFactors.map(row=>({feature:row.feature,impact:row.direction==="increases_risk"?"adverse":"supportive",shap_value:row.shap_value,raw_value:row.raw_value,explanation:row.reason_text})),
    formal_explanation: formalExplanation,
    full_explanation: `${formalExplanation} The calibrated default estimate is ${(pDefault * 100).toFixed(1)}%. The indicative model outcome is ${indicative}. Feature contributions below are SHAP attributions from the notebook's base LightGBM model.`,
    explanation_verified: false,
    recourse: null,
    model_version: model.model_version,
  };
}

async function loadClientModel(url = "runtime/model.json") {
  const response = await fetch(url, { cache: "force-cache" });
  if (!response.ok) throw new Error(`Could not load browser model artifacts (${response.status}).`);
  const model = await response.json();
  if (model.format_version !== 1 || model.classes.join(",") !== "0,1") {
    throw new Error("Browser model artifact format is unsupported.");
  }
  return model;
}

globalThis.CreditViewInference = {
  scoreApplicant,
  scoreTransformed,
  formFeatures,
  decideProbability,
  interventionalShap,
  loadClientModel,
};
