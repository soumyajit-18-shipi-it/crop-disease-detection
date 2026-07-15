const state = { groups: [], index: 0, status: "pending", search: "" };
const $ = (id) => document.getElementById(id);

async function loadQueue() {
  const response = await fetch(`/api/queue?status=${encodeURIComponent(state.status)}&search=${encodeURIComponent(state.search)}`);
  const data = await response.json();
  state.groups = data.groups;
  state.index = Math.min(state.index, Math.max(0, state.groups.length - 1));
  $("canonical-labels").innerHTML = data.canonical_labels.map(label => `<option value="${escapeHtml(label)}"></option>`).join("");
  renderSummary(data.summary);
  render();
}

function renderSummary(summary) {
  $("pending").textContent = summary.pending_groups.toLocaleString();
  $("validated").textContent = summary.validated_records.toLocaleString();
  $("rejected").textContent = summary.rejected_groups.toLocaleString();
  $("total").textContent = summary.total_records.toLocaleString();
}

function render() {
  const group = state.groups[state.index];
  $("review-card").hidden = !group;
  $("empty").hidden = Boolean(group);
  $("position").textContent = group ? `${state.index + 1} of ${state.groups.length} groups · ${group.record_count} records · priority ${group.priority}` : "No groups in this queue";
  $("previous").disabled = !group || state.index === 0;
  $("next").disabled = !group || state.index >= state.groups.length - 1;
  if (!group) return;
  $("crop").textContent = group.canonical_crop ? `${group.crop} → ${group.canonical_crop}` : group.crop || "Missing";
  $("original").textContent = group.original_disease || "Missing";
  $("suggested").textContent = group.suggested_disease || "No reliable suggestion";
  $("confidence").textContent = group.confidence == null ? "Not available" : `${Math.round(group.confidence * 100)}%`;
  $("symptoms").textContent = group.symptoms.length ? group.symptoms.join("; ") : "Not recorded";
  $("reasons").innerHTML = group.review_reasons.length ? group.review_reasons.map(reason => `<span class="flag">${escapeHtml(reason.replaceAll("_", " "))}</span>`).join("") : `<span class="flag">human confirmation required</span>`;
  $("images").innerHTML = group.image_urls.length ? group.image_urls.map((url, index) => `<img src="${url}" alt="Survey image ${index + 1}" loading="lazy">`).join("") : `<div class="flag">No available image</div>`;
  $("replacement").value = "";
  $("note").value = "";
  $("accept").disabled = !group.suggested_disease;
  $("decision-state").textContent = group.decision ? `Latest decision: ${group.decision.action} by ${group.decision.reviewer}` : "Pending human decision";
}

async function decide(action) {
  const group = state.groups[state.index];
  if (!group) return;
  const payload = { group_id: group.group_id, action, reviewer: $("reviewer").value, canonical_disease: $("replacement").value, note: $("note").value };
  if (action === "replace" && !payload.canonical_disease.trim()) { showToast("Enter a canonical replacement first."); return; }
  const response = await fetch("/api/decisions", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(payload) });
  const data = await response.json();
  if (!response.ok) { showToast(data.error || "Decision could not be saved."); return; }
  localStorage.setItem("fieldSurveyReviewer", payload.reviewer);
  showToast("Decision saved and validated manifest updated.");
  await loadQueue();
}

function showToast(message) { const toast = $("toast"); toast.textContent = message; toast.classList.add("show"); setTimeout(() => toast.classList.remove("show"), 2600); }
function escapeHtml(value) { return String(value).replace(/[&<>'"]/g, char => ({"&":"&amp;","<":"&lt;",">":"&gt;","'":"&#39;",'"':"&quot;"}[char])); }

document.querySelectorAll("[data-status]").forEach(button => button.addEventListener("click", () => {
  document.querySelectorAll("[data-status]").forEach(item => item.classList.remove("active"));
  button.classList.add("active"); state.status = button.dataset.status; state.index = 0; loadQueue();
}));
let searchTimer;
$("search").addEventListener("input", event => { clearTimeout(searchTimer); searchTimer = setTimeout(() => { state.search = event.target.value; state.index = 0; loadQueue(); }, 250); });
$("previous").addEventListener("click", () => { if (state.index > 0) { state.index--; render(); } });
$("next").addEventListener("click", () => { if (state.index < state.groups.length - 1) { state.index++; render(); } });
$("accept").addEventListener("click", () => decide("accept"));
$("reject").addEventListener("click", () => decide("reject"));
$("replace").addEventListener("click", () => decide("replace"));
$("reviewer").value = localStorage.getItem("fieldSurveyReviewer") || "";
loadQueue().catch(() => showToast("Could not load the review queue."));
