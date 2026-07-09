/* ── MediPredict frontend ────────────────────────────────────────────── */
(function () {
  "use strict";

  // ── Symptom categories ─────────────────────────────────────────────────
  const CATEGORIES = {
    fever:       ["fever", "high_fever", "mild_fever", "prolonged_fever", "chills", "sweating"],
    respiratory: ["cough", "wheezing", "shortness_of_breath", "difficulty_breathing",
                  "rapid_breathing", "chest_tightness", "bluish_lips", "runny_nose",
                  "nasal_congestion", "sneezing"],
    pain:        ["headache", "severe_headache", "chest_pain", "abdominal_pain",
                  "abdominal_cramps", "joint_pain", "body_aches", "eye_pain",
                  "sore_throat", "tingling_hands_feet"],
    digestive:   ["nausea", "vomiting", "diarrhea", "constipation", "loss_of_appetite",
                  "dehydration", "jaundice"],
  };

  function getCat(sym) {
    for (const [cat, list] of Object.entries(CATEGORIES)) {
      if (list.includes(sym)) return cat;
    }
    return "other";
  }

  // ── State ──────────────────────────────────────────────────────────────
  let selectedSymptoms = new Set();
  let activeModel = "random_forest";
  let activeCategory = "all";

  // ── DOM refs ───────────────────────────────────────────────────────────
  const predictBtn      = document.getElementById("predictBtn");
  const selectedCount   = document.getElementById("selectedCount");
  const symptomSearch   = document.getElementById("symptomSearch");
  const clearAllBtn     = document.getElementById("clearAll");
  const symptomGrid     = document.getElementById("symptomGrid");
  const resultPlaceholder = document.getElementById("resultPlaceholder");
  const resultContent   = document.getElementById("resultContent");
  const resultLoading   = document.getElementById("resultLoading");

  // ── Symptom chips ──────────────────────────────────────────────────────
  document.querySelectorAll(".symptom-chip").forEach((chip) => {
    const sym = chip.dataset.symptom;
    chip.addEventListener("click", () => {
      const cb = chip.querySelector("input");
      cb.checked = !cb.checked;
      chip.classList.toggle("checked", cb.checked);
      if (cb.checked) selectedSymptoms.add(sym);
      else selectedSymptoms.delete(sym);
      updateUI();
    });
  });

  // ── Search ─────────────────────────────────────────────────────────────
  symptomSearch.addEventListener("input", () => {
    const q = symptomSearch.value.toLowerCase().trim();
    filterChips(q, activeCategory);
  });

  // ── Category filter ────────────────────────────────────────────────────
  document.querySelectorAll(".cat-btn").forEach((btn) => {
    btn.addEventListener("click", () => {
      document.querySelectorAll(".cat-btn").forEach((b) => b.classList.remove("active"));
      btn.classList.add("active");
      activeCategory = btn.dataset.cat;
      filterChips(symptomSearch.value.toLowerCase().trim(), activeCategory);
    });
  });

  function filterChips(query, cat) {
    document.querySelectorAll(".symptom-chip").forEach((chip) => {
      const sym = chip.dataset.symptom;
      const label = sym.replace(/_/g, " ");
      const textMatch = !query || label.includes(query);
      const catMatch  = cat === "all" || getCat(sym) === cat;
      chip.classList.toggle("hidden", !(textMatch && catMatch));
    });
  }

  // ── Clear all ──────────────────────────────────────────────────────────
  clearAllBtn.addEventListener("click", () => {
    selectedSymptoms.clear();
    document.querySelectorAll(".symptom-chip").forEach((chip) => {
      chip.classList.remove("checked");
      chip.querySelector("input").checked = false;
    });
    updateUI();
    showPlaceholder();
  });

  // ── Model toggle ───────────────────────────────────────────────────────
  document.querySelectorAll(".model-btn").forEach((btn) => {
    btn.addEventListener("click", () => {
      document.querySelectorAll(".model-btn").forEach((b) => b.classList.remove("active"));
      btn.classList.add("active");
      activeModel = btn.dataset.model;
    });
  });

  // ── Tab navigation ─────────────────────────────────────────────────────
  document.querySelectorAll(".nav-tab").forEach((tab) => {
    tab.addEventListener("click", () => {
      const target = tab.dataset.tab;
      document.querySelectorAll(".nav-tab").forEach((t) => t.classList.remove("active"));
      document.querySelectorAll(".tab-content").forEach((c) => c.classList.remove("active"));
      tab.classList.add("active");
      document.getElementById(`tab-${target}`).classList.add("active");
    });
  });

  // ── Update shared UI ───────────────────────────────────────────────────
  function updateUI() {
    const n = selectedSymptoms.size;
    selectedCount.textContent = `${n} selected`;
    predictBtn.disabled = n === 0;
  }

  function showPlaceholder() {
    resultPlaceholder.style.display = "";
    resultContent.style.display = "none";
    resultLoading.style.display = "none";
  }

  // ── Predict ────────────────────────────────────────────────────────────
  predictBtn.addEventListener("click", async () => {
    if (selectedSymptoms.size === 0) return;

    resultPlaceholder.style.display = "none";
    resultContent.style.display = "none";
    resultLoading.style.display = "flex";

    try {
      const res = await fetch("/predict", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          symptoms: [...selectedSymptoms],
          model: activeModel,
        }),
      });
      const data = await res.json();
      if (data.error) throw new Error(data.error);
      renderResult(data);
    } catch (err) {
      resultLoading.style.display = "none";
      resultPlaceholder.style.display = "flex";
      resultPlaceholder.querySelector("p").textContent = "Error: " + err.message;
    }
  });

  // ── Render result ──────────────────────────────────────────────────────
  function renderResult(d) {
    resultLoading.style.display = "none";
    resultContent.style.display = "block";

    document.getElementById("resultDisease").textContent = d.prediction;

    // Confidence ring
    const pct   = d.confidence;
    const circ  = 2 * Math.PI * 34;
    const offset = circ - (pct / 100) * circ;
    const ring  = document.getElementById("ringFill");
    ring.style.strokeDashoffset = circ; // reset
    requestAnimationFrame(() => {
      ring.style.transition = "stroke-dashoffset 1s ease";
      ring.style.strokeDashoffset = offset;
      // colour by confidence
      ring.style.stroke = pct >= 70 ? "var(--green)" : pct >= 40 ? "var(--yellow)" : "var(--red)";
    });
    document.getElementById("ringLabel").textContent = pct + "%";

    // Meta
    const modelLabel = d.model_used === "random_forest" ? "Random Forest" : "Logistic Regression";
    document.getElementById("resultMeta").innerHTML =
      `<span>Model: ${modelLabel}</span><span>Symptoms provided: ${d.symptoms_provided}</span>`;

    // Top 3
    const top3El = document.getElementById("top3List");
    top3El.innerHTML = d.top3
      .map((item, i) => {
        const bw = item.confidence.toFixed(0) + "%";
        return `
        <div class="top3-item" style="--bar-w:${bw}">
          <span class="top3-rank">#${i + 1}</span>
          <span class="top3-name">${item.disease}</span>
          <span class="top3-conf">${item.confidence}%</span>
        </div>`;
      })
      .join("");

    // Symptom analysis
    const matchedEl = document.getElementById("matchedSymptoms");
    let html = "";
    if (d.matched_symptoms.length) {
      html += `<p style="font-size:.78rem;color:var(--text3);margin-bottom:6px">Matching ${d.prediction} symptoms:</p>`;
      html += `<div class="symptom-tags">` +
        d.matched_symptoms.map((s) => `<span class="sym-tag match">${fmt(s)}</span>`).join("") +
        `</div>`;
    }
    if (d.unmatched_symptoms.length) {
      html += `<p style="font-size:.78rem;color:var(--text3);margin:8px 0 6px">Unrelated symptoms:</p>`;
      html += `<div class="symptom-tags">` +
        d.unmatched_symptoms.map((s) => `<span class="sym-tag nomatch">${fmt(s)}</span>`).join("") +
        `</div>`;
    }
    matchedEl.innerHTML = html;
  }

  function fmt(sym) {
    return sym.replace(/_/g, " ").replace(/\b\w/g, (c) => c.toUpperCase());
  }

  // ── Init ───────────────────────────────────────────────────────────────
  updateUI();
})();
