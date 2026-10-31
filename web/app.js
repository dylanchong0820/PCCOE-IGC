/* S.E.D.O.S.S. farmer hub. Talks to the unit's API (the virtual Pi today, the real Pi later). */
(() => {
  "use strict";

  // Same origin when the Pi serves this page; otherwise the local virtual Pi (override with ?api=).
  const API = new URLSearchParams(location.search).get("api")
    || (location.protocol.startsWith("http") && location.port === "8765" ? "" : "http://127.0.0.1:8765");

  const $ = (id) => document.getElementById(id);
  const MONTHS = ["January", "February", "March", "April", "May", "June", "July", "August", "September", "October", "November", "December"];
  const DAYS = ["Sunday", "Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday"];
  const DECISION_LABEL = {
    irrigate: "Watered", irrigate_heatwave: "Watered (heatwave, evening)",
    rain_delay: "Rain delay", no_irrigation_needed: "Not needed",
  };
  const FAR_KM = 50;

  let S = null;              // latest unit state
  let events = [];
  let daily = [];
  let sitesIndex = null;
  let hardwareLoaded = false;
  let leafDate = null;
  let failures = 0;
  let view = "today";
  let pendingConfirm = null;
  let alertFilter = "all";
  let map = null;
  let markers = {};
  let clickMarker = null;
  let rosterKey = null;      // redraw the site list only when the site or search changes

  // ---------- helpers ----------
  const esc = (v) => String(v ?? "").replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
  const num = (v, d = 1) => (v === null || v === undefined || Number.isNaN(+v) ? "–" : (+v).toFixed(d));
  const parseT = (iso) => new Date(iso);   // the unit reports local (orchard) time without a zone
  const hhmm = (iso) => { const d = parseT(iso); return `${String(d.getHours()).padStart(2, "0")}:${String(d.getMinutes()).padStart(2, "0")}`; };
  const dayLabel = (iso) => { const d = parseT(iso); return `${d.getDate()} ${MONTHS[d.getMonth()].slice(0, 3)}`; };

  async function api(path, body) {
    const opts = body === undefined ? {} : { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(body) };
    const r = await fetch(API + path, opts);
    const data = await r.json().catch(() => ({}));
    if (!r.ok) throw new Error(data.error || `The unit answered ${r.status}`);
    return data;
  }

  function feedback(msg, isErr = false) {
    const f = $("feedback");
    f.textContent = msg;
    f.classList.toggle("err", isErr);
  }

  async function send(cmd, args, okMsg) {
    try {
      await api("/api/command", { cmd, args });
      feedback(okMsg);
      await refresh();
    } catch (e) {
      feedback(`Could not send that to the unit: ${e.message}`, true);
    }
  }

  function haversine(a1, o1, a2, o2) {
    const r = Math.PI / 180, R = 6371;
    const da = (a2 - a1) * r, dl = (o2 - o1) * r;
    const h = Math.sin(da / 2) ** 2 + Math.cos(a1 * r) * Math.cos(a2 * r) * Math.sin(dl / 2) ** 2;
    return 2 * R * Math.asin(Math.sqrt(h));
  }

  // ---------- views ----------
  function showView(name) {
    if (!document.querySelector(`.view[data-view="${name}"]`)) name = "today";
    view = name;
    document.querySelectorAll(".view").forEach((v) => { v.hidden = v.dataset.view !== name; });
    document.querySelectorAll(".nav a").forEach((a) => {
      if (a.dataset.view === name) a.setAttribute("aria-current", "page"); else a.removeAttribute("aria-current");
    });
    if (name === "sites") { initMap(); pollJobs(); }
    if (name === "history") loadDaily();
    if (name === "bench") loadHardware();
    renderView();
  }
  window.addEventListener("hashchange", () => showView(location.hash.slice(1)));

  function renderView() {
    if (!S) return;
    if (view === "today") renderToday();
    if (view === "bench") renderBench();
    if (view === "sites" && rosterKey !== S.site.id + $("site-search").value) renderSites();
    if (view === "history") renderHistory();
    if (view === "alerts") renderAlerts();
  }

  // ---------- polling ----------
  async function refresh() {
    try {
      const [state, ev] = await Promise.all([api("/api/state"), api("/api/events?limit=300")]);
      failures = 0;
      $("offline").hidden = true;
      const dayChanged = S && S.time.slice(0, 10) !== state.time.slice(0, 10);
      S = state;
      events = ev;
      renderRail();
      if (dayChanged && view === "history") loadDaily();
      renderView();
    } catch (e) {
      failures += 1;
      if (failures >= 2) $("offline").hidden = false;
    }
  }

  function renderRail() {
    $("rail-site").textContent = S.site.name;
    $("rail-site-meta").textContent = `${num(S.site.latitude, 2)}°N ${num(S.site.longitude, 2)}°E`;
    const start = parseT(`${S.replay.start}T00:00:00`), now = parseT(S.time), end = parseT(`${S.replay.end}T00:00:00`);
    const day = Math.floor((now - start) / 86400000) + 1, total = Math.round((end - start) / 86400000) + 1;
    $("rail-replay").textContent = S.finished
      ? `Weather replay finished (${total} days)`
      : `Replaying ${S.replay.start.slice(0, 4)} weather · day ${Math.min(day, total)} of ${total}`;
    const sel = $("speed");
    if (document.activeElement !== sel) {
      if (![...sel.options].some((o) => o.value === String(S.speed))) {
        const o = document.createElement("option");     // a speed set from elsewhere (API or script)
        o.value = String(S.speed);
        o.textContent = `${S.speed}× real time`;
        sel.append(o);
      }
      sel.value = String(S.speed);
    }
    const attention = openAlerts().length;
    const c = $("alert-count");
    c.hidden = attention === 0;
    c.textContent = attention;
  }

  // Alerts that need the farmer's attention: today's warnings and alarms (same scope everywhere).
  function openAlerts() {
    const today = S.time.slice(0, 10);
    return events.filter((e) => e.t.slice(0, 10) === today && (e.level === "alarm" || e.level === "warn"));
  }

  // ---------- today ----------
  function todaysDecision() {
    const d = S.last_decision;
    return d && d.date === S.time.slice(0, 10) ? d : null;
  }

  function todaysDelivery() {
    const d = S.last_delivery;
    return d && d.started.slice(0, 10) === S.time.slice(0, 10) ? d : null;
  }

  function doAvoid(dec) {
    const ov = S.farmer_overrides_today;
    if (!dec) {
      return { sub: "Decision at 05:00", do: ["Nothing yet. The unit decides at <b>05:00</b>."], avoid: ["Nothing yet."] };
    }
    const rec = dec.recipe;
    const inp = dec.sensor_and_forecast_inputs;
    if (ov.skip) {
      return { sub: "Skipped by you", do: ["Nothing. You skipped today's watering."], avoid: ["Watering until tomorrow's 05:00 decision."] };
    }
    if (rec.scenario === "rain_delay") {
      return {
        sub: "Rain expected, no watering",
        do: [`Let the rain water the trees (<b>${num(inp.rain_forecast_mm)} mm</b> forecast).`],
        avoid: ["Irrigating today.", "Fertilizer: rain would wash it off."],
      };
    }
    if (!rec.irrigate) {
      return {
        sub: "No watering needed",
        do: [`Nothing. The soil holds <b>${num(inp.soil_water_mm, 0)} mm</b> of water, above the stress line.`],
        avoid: ["Watering today: the water would drain past the roots."],
      };
    }
    const plan = S.today_plan, done = todaysDelivery();
    const when = plan ? hhmm(plan.start) : done ? hhmm(done.started) : rec.timing;
    const litres = plan ? plan.volume_l : done ? done.volume_l : null;
    const setpoint = dec.applied_ec_setpoint_dS_m;
    const d = [`Irrigate <b>${num(rec.volume_mm, 0)} mm</b>${litres ? ` (${num(litres)} L on the test bed)` : ""} at <b>${esc(when)}</b>.`];
    d.push(setpoint && !ov.water_only ? `Add nutrients to <b>EC ${num(setpoint, 2)} dS/m</b>.` : "Water only, no nutrients.");
    const a = [];
    if (rec.scenario === "irrigate_heatwave") {
      a.push(`Watering in the midday heat (forecast top ${num(inp.tmax_forecast_C)} °C).`);
      if (setpoint && !ov.water_only) a.push("A full nutrient dose: halved for the heat.");
    }
    if (setpoint && rec.ec_setpoint_dS_m > setpoint && !ov.water_only) a.push(`Nutrients above <b>EC ${num(setpoint, 2)}</b>, the crop-safe salt limit.`);
    if (!a.length) a.push("Watering again later today.");
    return { sub: rec.scenario === "irrigate_heatwave" ? "Heatwave: water this evening" : "Water this morning", do: d, avoid: a };
  }

  function renderLeaf() {
    const now = parseT(S.time);
    const key = S.site.id + S.time.slice(0, 10);
    if (leafDate && leafDate !== key && leafDate.slice(0, -10) === S.site.id) tearOff();
    leafDate = key;
    $("leaf-month").textContent = `${MONTHS[now.getMonth()]} ${now.getFullYear()}`;
    $("leaf-weekday").textContent = DAYS[now.getDay()];
    $("leaf-day").textContent = now.getDate();

    const dec = todaysDecision();
    const da = doAvoid(dec);
    $("leaf-sub").textContent = S.paused ? "Unit paused" : da.sub;
    $("do-list").innerHTML = da.do.map((x) => `<li>${x}</li>`).join("");
    $("avoid-list").innerHTML = da.avoid.map((x) => `<li>${x}</li>`).join("");

    if (dec) {
      const inp = dec.sensor_and_forecast_inputs, mo = dec.model_outputs;
      const thr = S.simulation_truth.stress_threshold_mm;
      const ch = mo.predicted_soil_water_change_mm;
      const move = Math.abs(ch) < 0.5 ? "stay about the same" : ch < 0 ? `drop about ${num(-ch, 0)} mm` : `rise about ${num(ch, 0)} mm`;
      $("why").innerHTML =
        `<b>Why:</b> the soil holds <b>${num(inp.soil_water_mm, 0)} mm</b> of water (stress line ${num(thr, 0)} mm). ` +
        `The trees will use about <b>${num(mo.forecast_et0_mm_per_day)} mm</b> today and the soil will ${move}. ` +
        `Forecast: ${num(inp.rain_forecast_mm)} mm rain, top ${num(inp.tmax_forecast_C, 0)} °C.`;
    } else {
      $("why").textContent = "";
    }
    renderPlan(dec);
  }

  function renderPlan(dec) {
    const p = $("plan");
    if (S.paused) { p.innerHTML = `<span class="state">Unit paused.</span> Pumps and dosing are stopped until you resume.`; return; }
    if (S.delivery) {
      const d = S.delivery, pct = Math.min(100, (d.delivered_l / d.volume_l) * 100);
      p.innerHTML = `<span class="state">Watering now:</span> ${num(d.delivered_l, 2)} of ${num(d.volume_l, 2)} L` +
        (d.ec_setpoint_dS_m ? `, nutrients at EC ${num(S.sensors.final_ec_dS_m, 2)} (target ${num(d.ec_setpoint_dS_m, 2)})` : ", water only") +
        `<div class="bar" role="progressbar" aria-valuemin="0" aria-valuemax="100" aria-valuenow="${pct.toFixed(0)}" aria-label="Watering progress"><i style="width:${pct}%"></i></div>`;
      return;
    }
    if (S.today_plan) {
      const t = S.today_plan;
      p.innerHTML = `<span class="state">Scheduled ${esc(hhmm(t.start))}:</span> ${num(t.volume_l, 2)} L` +
        (t.ec_setpoint_dS_m ? ` with nutrients to EC ${num(t.ec_setpoint_dS_m, 2)}` : ", water only");
      return;
    }
    const done = todaysDelivery();
    if (done) {
      p.innerHTML = `<span class="state">Done ${esc(hhmm(done.finished))}:</span> ${num(done.volume_l, 2)} L delivered` +
        (done.final_ec_mean_dS_m ? ` at EC ${num(done.final_ec_mean_dS_m, 2)} (settled in ${done.settle_s} s)` : "");
      return;
    }
    p.innerHTML = dec ? `<span class="state">No watering scheduled today.</span>` : "";
  }

  function tearOff() {
    const leaf = $("leaf");
    const ghost = leaf.cloneNode(true);
    ghost.removeAttribute("id");
    ghost.querySelectorAll("[id]").forEach((n) => n.removeAttribute("id"));
    ghost.setAttribute("aria-hidden", "true");
    ghost.classList.add("tearing");
    $("leaf-stack").appendChild(ghost);
    ghost.addEventListener("animationend", () => ghost.remove(), { once: true });
    setTimeout(() => ghost.remove(), 1600);
  }

  function renderReadings() {
    const s = S.sensors, a = S.actuators, t = S.tanks_l;
    $("r-feed").textContent = num(s.feed_ec_dS_m, 2);
    const treating = a.fdi_mode === "treat", lastBatch = S.fdi_last_batch_ec_dS_m;
    $("r-outlet").textContent = treating || lastBatch == null ? num(s.outlet_ec_dS_m, 2) : num(lastBatch, 2);
    $("r-fdi").textContent = treating ? "desalinating now" : a.fdi_mode === "regen" ? "cleaning electrodes" : lastBatch == null ? "idle" : "last batch · idle";
    $("r-outlet").parentElement.classList.toggle("stale", a.fdi_mode === "regen");
    $("r-final").textContent = num(s.final_ec_dS_m, 2);
    const limit = S.crop_safe_ec_limit_dS_m;
    const finalCell = $("r-final").parentElement;
    const over = s.final_ec_dS_m > limit + 0.05;
    finalCell.classList.toggle("bad", over);
    $("r-final-note").textContent = `${S.delivery ? "live" : "last water"} · limit ${num(limit, 2)} ±0.05${over ? " · over" : ""}`;
    $("r-buffer").textContent = num(t.buffer, 1);
    $("r-brine").textContent = num(t.brine, 1);
    $("r-salt").textContent = num(S.salt_recovered_g, 1);
    $("clock").textContent = `${hhmm(S.time)} at ${S.site.name}`;
  }

  function friendly(msg) {
    const m = msg.match(/EC setpoint ([\d.]+) dS\/m is above the crop-safe limit ([\d.]+) dS\/m/);
    if (m) return `<b>Safety cap:</b> the AI asked for nutrients to EC ${m[1]} dS/m. The unit held it at the crop-safe ${m[2]} dS/m.`;
    if (msg.startsWith("AI decision:")) {
      const soil = msg.match(/soil water (\d+) mm/), thr = msg.match(/threshold (\d+) mm/), rain = msg.match(/forecast ([\d.]+) mm/);
      if (msg.includes("RAIN DELAY")) return `<b>AI decision:</b> no watering, ${rain ? `${rain[1]} mm of rain` : "rain"} forecast.`;
      if (msg.includes("no irrigation needed")) return `<b>AI decision:</b> no watering needed. Soil will stay near ${soil ? soil[1] : "?"} mm, above the ${thr ? thr[1] : "?"} mm stress line.`;
      if (msg.includes("-> irrigate")) {
        const heat = msg.includes("HEATWAVE") ? " Heatwave: water in the evening, half the nutrients." : "";
        return `<b>AI decision:</b> water today. Soil would fall to ${soil ? soil[1] : "?"} mm, under the ${thr ? thr[1] : "?"} mm stress line.${heat}`;
      }
    }
    if (msg.startsWith("Electrodes saturated")) return "Electrodes full: salty water sent to the brine tank while they clean.";
    if (msg === "FDI regeneration finished") return "Electrode cleaning finished.";
    const dl = msg.match(/^Delivering ([\d.]+) L(, dosing to EC ([\d.]+) dS\/m)?/);
    if (dl) return `Watering started: ${dl[1]} L${dl[3] ? ` with nutrients to EC ${dl[3]}` : ", water only"}.`;
    const df = msg.match(/^Delivery finished: ([\d.]+) L/);
    if (df) return `Watering finished: ${df[1]} L.`;
    if (msg.startsWith("Command: ")) return `You sent: ${esc(msg.slice(9).replaceAll("_", " "))}`;
    return esc(msg);
  }

  function renderNotice() {
    const hit = openAlerts().slice(-1)[0];
    const n = $("notice");
    if (!hit) { n.hidden = true; return; }
    n.hidden = false;
    n.className = `notice${hit.level === "alarm" ? " alarm" : ""}`;
    n.innerHTML = `${esc(hhmm(hit.t))} · ${friendly(hit.msg)}`;
  }

  function renderControls() {
    const ov = S.farmer_overrides_today, dec = todaysDecision();
    const pause = $("btn-pause");
    pause.textContent = S.paused ? "Resume unit" : "Pause unit";
    pause.classList.toggle("active", S.paused);
    const skip = $("btn-skip"), wo = $("btn-water-only");
    skip.textContent = ov.skip ? "Skipped today" : "Skip today";
    skip.disabled = ov.skip || S.finished;
    wo.textContent = ov.water_only ? "Water only: on" : "Water only today";
    wo.disabled = ov.water_only || ov.skip || S.finished;
    $("btn-irrigate").disabled = !!S.delivery || S.finished;
    if (!dec && !ov.skip) wo.title = "Takes effect for today's 05:00 decision";
  }

  function renderTodayLog() {
    const today = S.time.slice(0, 10);
    const items = events.filter((e) => e.t.slice(0, 10) === today && e.source !== "sim").slice(-12).reverse();
    $("today-log").innerHTML = items.length
      ? items.map((e) => `<li class="lvl-${e.level}"><time>${esc(hhmm(e.t))}</time><span>${friendly(e.msg)}</span></li>`).join("")
      : `<li><span class="empty">Nothing logged yet today.</span></li>`;
  }

  function renderToday() {
    renderLeaf();
    renderReadings();
    renderNotice();
    renderControls();
    renderTodayLog();
  }

  // controls
  $("btn-irrigate").addEventListener("click", () => {
    const f = $("irrigate-form"), open = f.hidden;
    f.hidden = !open;
    $("btn-irrigate").setAttribute("aria-expanded", String(open));
    $("confirm").hidden = true;
    if (open) $("irr-mm").focus();
  });
  $("irr-cancel").addEventListener("click", () => { $("irrigate-form").hidden = true; $("btn-irrigate").setAttribute("aria-expanded", "false"); });
  $("irrigate-form").addEventListener("submit", (ev) => {
    ev.preventDefault();
    const mm = +$("irr-mm").value, fert = $("irr-fert").checked;
    if (!(mm >= 1 && mm <= 15)) { feedback("Choose between 1 and 15 mm.", true); return; }
    $("irrigate-form").hidden = true;
    $("btn-irrigate").setAttribute("aria-expanded", "false");
    send("irrigate_now", { mm, fertigate: fert }, `Watering ${mm} mm${fert ? " with nutrients" : ""} now.`);
  });
  function ask(text, cmd, okMsg) {
    pendingConfirm = { cmd, okMsg };
    $("confirm-text").textContent = text;
    $("confirm").hidden = false;
    $("irrigate-form").hidden = true;
    $("confirm-yes").focus();
  }
  $("btn-skip").addEventListener("click", () => ask("Skip all watering today? This cannot be undone until tomorrow.", "skip_today", "Today's watering is skipped."));
  $("btn-water-only").addEventListener("click", () => ask("Water without nutrients for the rest of today?", "water_only_today", "Nutrients are off for today."));
  $("confirm-yes").addEventListener("click", () => {
    $("confirm").hidden = true;
    if (pendingConfirm) send(pendingConfirm.cmd, {}, pendingConfirm.okMsg);
    pendingConfirm = null;
  });
  $("confirm-no").addEventListener("click", () => { $("confirm").hidden = true; pendingConfirm = null; });
  $("btn-pause").addEventListener("click", () => {
    if (S.paused) send("resume", {}, "Unit resumed."); else send("pause", {}, "Unit paused. Pumps and dosing stopped.");
  });
  $("speed").addEventListener("change", async (e) => {
    try { await api("/api/speed", { sim_seconds_per_second: +e.target.value }); } catch (err) { feedback(err.message, true); }
  });

  // ---------- unit ----------
  function renderBench() {
    const s = S.sensors, a = S.actuators, t = S.tanks_l;
    const st = (txt, cls = "") => `<span class="st ${cls}">${esc(txt)}</span>`;
    const stages = [
      { h: "Raw water", b: `${num(t.feed, 1)} L`, s: `salt EC ${num(s.feed_ec_dS_m, 2)} dS/m`, st: a.feed_pump ? st("Feed pump on", "on") : st("Feed pump off"), live: a.feed_pump },
      { h: "Desalination",
        b: `EC ${num(a.fdi_mode !== "treat" && S.fdi_last_batch_ec_dS_m != null ? S.fdi_last_batch_ec_dS_m : s.outlet_ec_dS_m, 2)}`,
        s: a.fdi_mode === "treat" ? `${num(s.feed_flow_l_min, 2)} L/min through the FDI cell` : a.fdi_mode === "regen" ? "cell rinsing salt to brine" : "last treated batch",
        st: a.fdi_mode === "treat" ? st("Desalinating", "on") : a.fdi_mode === "regen" ? st("Cleaning electrodes", "regen") : st("Idle"), live: a.fdi_mode !== "idle" },
      { h: "3-way valve", b: a.three_way_valve === "buffer" ? "To clean tank" : "To brine", s: `${S.fdi_regenerations} electrode cleanings so far`,
        st: a.three_way_valve === "buffer" ? st("Normal") : st("Diverting salty water", "regen"), live: a.fdi_mode !== "idle" },
      { h: "Clean water", b: `${num(t.buffer, 1)} L`, s: s.buffer_high_float ? "full (high float)" : s.buffer_low_float ? "low (low float)" : "between floats",
        st: s.buffer_low_float ? st("Low", "warn") : st("OK", "on"), live: S.actuators.delivery_pump },
      { h: "Nutrient dosing", b: `EC ${num(s.final_ec_dS_m, 2)}`, s: `pumps A/B ${num(a.dosing_pump_a_ml_min, 1)} mL/min each · pH ${num(s.ph, 2)}`,
        st: a.dosing_pump_a_ml_min > 0 ? st("Dosing", "on") : st("Off"), live: a.delivery_pump },
      { h: "Test bed", b: `VWC ${num(s.soil_vwc, 3)}`, s: `soil probe raw ${s.soil_probe_raw}`, st: a.delivery_pump ? st("Watering", "on") : st("Resting"), live: false },
    ];
    $("flow").innerHTML = stages.map((x) => `<div class="stage${x.live ? " live" : ""}"><h3>${esc(x.h)}</h3><b>${esc(x.b)}</b><small>${esc(x.s)}</small>${x.st}</div>`).join("");

    const lv = (v, cap, cls = "") => `<span class="lv ${cls}${v / cap < 0.15 ? " low" : ""}"><i style="width:${Math.min(100, (v / cap) * 100)}%"></i></span>`;
    $("tanks").innerHTML = `<tr><th>Tank</th><th>Level</th><th class="n">Litres</th></tr>
      <tr><td>Raw water</td><td>${lv(t.feed, 40)}</td><td class="n">${num(t.feed, 1)} / 40</td></tr>
      <tr><td>Clean water</td><td>${lv(t.buffer, 15)}</td><td class="n">${num(t.buffer, 1)} / 15</td></tr>
      <tr><td>Brine</td><td>${lv(t.brine, 15, "brine")}</td><td class="n">${num(t.brine, 1)} / 15</td></tr>
      <tr><td>Evaporation trays</td><td>${lv(t.evaporation_trays, 3, "brine")}</td><td class="n">${num(t.evaporation_trays, 2)} / 3</td></tr>
      <tr><td>Nutrient A</td><td>${lv(t.nutrient_a, 2)}</td><td class="n">${num(t.nutrient_a, 2)} / 2</td></tr>
      <tr><td>Nutrient B</td><td>${lv(t.nutrient_b, 2)}</td><td class="n">${num(t.nutrient_b, 2)} / 2</td></tr>`;
    $("sensors").innerHTML = `<tr><th>Sensor</th><th class="n">Reading</th></tr>
      <tr><td>Air temperature</td><td class="n">${num(s.air_temp_c)} °C</td></tr>
      <tr><td>Humidity</td><td class="n">${num(s.rh_pct, 0)} %</td></tr>
      <tr><td>Sunlight</td><td class="n">${num(s.solar_w_m2, 0)} W/m²</td></tr>
      <tr><td>Water temperature</td><td class="n">${num(s.water_temp_c)} °C</td></tr>
      <tr><td>pH (after dosing)</td><td class="n">${num(s.ph, 2)}</td></tr>
      <tr><td>Soil probe</td><td class="n">${s.soil_probe_raw} counts · VWC ${num(s.soil_vwc, 3)}</td></tr>`;
  }

  async function loadHardware() {
    if (hardwareLoaded) return;
    try {
      const parts = await api("/api/hardware");
      $("parts").innerHTML = `<tr><th>Part</th><th>Job in the unit</th></tr>` +
        parts.map((p) => `<tr><td>${esc(p.part)}</td><td>${esc(p.role)}</td></tr>`).join("");
      hardwareLoaded = true;
    } catch (e) { /* retried next time the view opens */ }
  }
  document.querySelectorAll("[data-cmd]").forEach((b) => b.addEventListener("click", () => {
    const labels = { regenerate_fdi: "Electrode cleaning started.", refill_feed: "Raw-water tank marked full.", refill_nutrients: "Nutrient stock marked full." };
    send(b.dataset.cmd, {}, labels[b.dataset.cmd]);
  }));

  // ---------- sites ----------
  async function loadSites() {
    if (!sitesIndex) sitesIndex = await api("/api/sites");
    return sitesIndex;
  }

  function siteFacts(s) {
    const m = s.metrics.et0_forecast["XGBoost - observations only"];
    return `<dl><dt>Location</dt><dd>${num(s.latitude, 2)}°N ${num(s.longitude, 2)}°E</dd>
      <dt>Weather data</dt><dd>${esc(s.data_range[0].slice(0, 4))}–${esc(s.data_range[1].slice(0, 4))}, NASA POWER</dd>
      <dt>Heatwave from</dt><dd>${num(s.heat_tmax_c)} °C</dd>
      <dt>Water-use forecast</dt><dd>±${num(m.RMSE, 2)} mm/day error</dd></dl>`;
  }

  async function renderSites() {
    let idx;
    try { idx = await loadSites(); } catch (e) { return; }
    rosterKey = S.site.id + $("site-search").value;
    const q = $("site-search").value.trim().toLowerCase();
    const list = idx.sites.filter((s) => !q || s.name.toLowerCase().includes(q) || s.id.includes(q));
    const cur = S.site.id;
    $("roster").innerHTML = list.length
      ? list.map((s) => `<li class="${s.id === cur ? "current" : ""}" data-site="${esc(s.id)}">
          <h3>${esc(s.name)}${s.id === cur ? "<em>In use</em>" : ""}</h3>${siteFacts(s)}
          ${s.id === cur ? "" : `<button class="btn" data-use="${esc(s.id)}">Use this site</button>`}</li>`).join("")
      : `<li class="none">No trained site matches “${esc($("site-search").value)}”. Click your orchard on the map to find the nearest one.</li>`;
    if (map) {
      Object.entries(markers).forEach(([id, mk]) => {
        mk.setIcon(L.divIcon({ className: "", html: `<div class="site-pin${id === cur ? " current" : ""}"></div>`, iconSize: [18, 18], iconAnchor: [9, 9] }));
      });
    }
  }

  async function initMap() {
    if (map || typeof L === "undefined") { if (map) setTimeout(() => map.invalidateSize(), 50); return; }
    let idx;
    try { idx = await loadSites(); } catch (e) { return; }
    map = L.map("map", { zoomControl: true, attributionControl: true }).setView([3.6, 102.1], 7);
    L.tileLayer("https://tile.openstreetmap.org/{z}/{x}/{y}.png", { maxZoom: 13, attribution: "© OpenStreetMap contributors" }).addTo(map);
    addMarkers(idx);
    map.on("click", (e) => showNearest(e.latlng.lat, e.latlng.lng));
    setTimeout(() => map.invalidateSize(), 50);
    renderSites();
  }

  function addMarkers(idx) {
    idx.sites.forEach((s) => {
      if (markers[s.id]) return;
      markers[s.id] = L.marker([s.latitude, s.longitude], {
        title: s.name, keyboard: true,
        icon: L.divIcon({ className: "", html: `<div class="site-pin"></div>`, iconSize: [18, 18], iconAnchor: [9, 9] }),
      }).addTo(map).bindTooltip(s.name, { direction: "top", offset: [0, -10] })
        .on("click", () => { $("site-search").value = s.name; renderSites(); });
    });
  }

  function showNearest(lat, lon) {
    const idx = sitesIndex;
    let best = null, bestD = Infinity;
    idx.sites.forEach((s) => { const d = haversine(lat, lon, s.latitude, s.longitude); if (d < bestD) { best = s; bestD = d; } });
    if (clickMarker) clickMarker.remove();
    clickMarker = L.marker([lat, lon], { icon: L.divIcon({ className: "", html: `<div class="click-pin"></div>`, iconSize: [12, 12], iconAnchor: [6, 6] }) }).addTo(map);
    const far = bestD > FAR_KM, n = $("nearest");
    n.hidden = false;
    n.classList.toggle("far", far);
    n.innerHTML = `<b>Your orchard: ${num(lat, 2)}°N ${num(lon, 2)}°E</b>
      <span>Nearest trained site: <b>${esc(best.name)}</b>, ${num(bestD, 0)} km away.</span>
      ${far ? `<span>That is beyond ${FAR_KM} km, so its weather may not match yours. Ask the team to train a site for your area before relying on it.</span>` : `<span>Close enough to share its weather.</span>`}
      ${best.id === S.site.id ? "<span>This site is already in use.</span>" : `<button class="btn" data-use="${esc(best.id)}">Use ${esc(best.name)}</button>`}
      <button class="btn ghost" type="button" data-add-here="${lat},${lon}">Add this spot as a new site</button>`;
    if (!$("add-site").hidden) { $("as-lat").value = lat.toFixed(4); $("as-lon").value = lon.toFixed(4); nearHint(); }
  }

  // ---------- add a site ----------
  let jobsTimer = null;
  const jobState = {};

  function openAddForm(lat, lon) {
    $("add-site").hidden = false;
    $("btn-add-site").setAttribute("aria-expanded", "true");
    if (lat !== undefined) { $("as-lat").value = (+lat).toFixed(4); $("as-lon").value = (+lon).toFixed(4); }
    nearHint();
    $("as-name").focus();
  }
  function closeAddForm() {
    $("add-site").hidden = true;
    $("btn-add-site").setAttribute("aria-expanded", "false");
    $("as-err").textContent = "";
  }

  function nearHint() {
    const lat = parseFloat($("as-lat").value), lon = parseFloat($("as-lon").value), el = $("as-near");
    if (Number.isNaN(lat) || Number.isNaN(lon) || !sitesIndex || !sitesIndex.sites.length) { el.textContent = ""; return; }
    let best = null, bestD = Infinity;
    sitesIndex.sites.forEach((s) => { const d = haversine(lat, lon, s.latitude, s.longitude); if (d < bestD) { best = s; bestD = d; } });
    const close = bestD < FAR_KM;
    el.classList.toggle("warn", close);
    el.textContent = close
      ? `Only ${num(bestD, 0)} km from ${best.name}. NASA weather comes on a grid about 50 km wide, so this site would likely get the same weather. You can still add it.`
      : `Nearest trained site: ${best.name}, ${num(bestD, 0)} km away.`;
  }

  $("btn-add-site").addEventListener("click", () => ($("add-site").hidden ? openAddForm() : closeAddForm()));
  $("as-cancel").addEventListener("click", closeAddForm);
  ["as-lat", "as-lon"].forEach((id) => $(id).addEventListener("input", nearHint));
  document.addEventListener("click", (e) => {
    const b = e.target.closest("[data-add-here]");
    if (!b) return;
    const [lat, lon] = b.dataset.addHere.split(",");
    openAddForm(lat, lon);
  });

  $("add-site").addEventListener("submit", async (ev) => {
    ev.preventDefault();
    const err = $("as-err");
    const name = $("as-name").value.trim(), lat = $("as-lat").value, lon = $("as-lon").value, elev = $("as-elev").value;
    if (name.length < 2) { err.textContent = "Give the site a name."; $("as-name").focus(); return; }
    if (lat === "" || lon === "") { err.textContent = "Enter the latitude and longitude, or click the orchard on the map."; $("as-lat").focus(); return; }
    err.textContent = "";
    const btn = ev.submitter || $("add-site").querySelector("[type=submit]");
    btn.disabled = true;
    try {
      await api("/api/sites", { name, latitude: +lat, longitude: +lon, elevation_m: elev === "" ? null : +elev });
      $("add-site").reset();
      $("as-near").textContent = "";
      closeAddForm();
      pollJobs();
    } catch (e) {
      err.textContent = e.message;
    } finally {
      btn.disabled = false;
    }
  });

  async function pollJobs() {
    clearTimeout(jobsTimer);
    let jobs;
    try { jobs = await api("/api/sites/jobs"); } catch (e) { jobsTimer = setTimeout(pollJobs, 4000); return; }
    let newlyDone = false;
    jobs.forEach((j) => { if (j.status === "done" && jobState[j.id] && jobState[j.id] !== "done") newlyDone = true; jobState[j.id] = j.status; });
    if (newlyDone) {
      sitesIndex = null;
      try { const idx = await loadSites(); if (map) addMarkers(idx); } catch (e) { /* next poll retries */ }
      rosterKey = null;
      renderSites();
    }
    const nowS = Date.now() / 1000;
    $("site-jobs").innerHTML = jobs.slice(0, 4).map((j) => {
      if (j.status === "queued" || j.status === "training") {
        const secs = Math.max(0, Math.round(nowS - (j.started_at || j.queued_at)));
        return `<li><b>${esc(j.name)}</b><span>${j.status === "queued" ? "Waiting for the previous site to finish…" : `Downloading weather and training models… ${secs} s`}</span><div class="bar" role="progressbar" aria-label="Training ${esc(j.name)}" aria-valuemin="0" aria-valuemax="100" aria-valuenow="${Math.min(95, Math.round(secs / 60 * 100))}"><i style="width:${j.status === "queued" ? 0 : Math.min(95, secs / 60 * 100)}%"></i></div></li>`;
      }
      if (j.status === "done") {
        const inUse = S && S.site.id === j.id;
        return `<li class="done"><b>${esc(j.name)} is ready</b><span>Trained on 2015–2024 NASA weather. It is on the map and in the list below.</span>${inUse ? "" : `<button class="btn" data-use="${esc(j.id)}">Use ${esc(j.name)}</button>`}</li>`;
      }
      return `<li class="failed"><b>${esc(j.name)} was not added</b><span>${esc(j.error)}</span></li>`;
    }).join("");
    if (jobs.some((j) => j.status === "queued" || j.status === "training")) jobsTimer = setTimeout(pollJobs, 1500);
  }

  document.addEventListener("click", async (e) => {
    const b = e.target.closest("[data-use]");
    if (!b) return;
    b.disabled = true;
    b.textContent = "Switching…";
    try {
      await api("/api/site", { site: b.dataset.use });
      leafDate = null;
      $("nearest").hidden = true;
      $("site-search").value = "";
      await refresh();
      daily = [];
      location.hash = "#today";
      feedback(`Unit restarted with ${S.site.name} models and weather.`);
    } catch (err) {
      b.disabled = false;
      b.textContent = "Try again";
      alert(`Could not switch site: ${err.message}`);
    }
  });
  $("site-search").addEventListener("input", renderSites);

  // ---------- history ----------
  async function loadDaily() {
    try { daily = await api("/api/daily"); renderHistory(); } catch (e) { /* offline state handles it */ }
  }

  function renderHistory() {
    const rows = daily;
    const thr = S.simulation_truth.stress_threshold_mm;
    if (!rows.length) {
      $("chart").innerHTML = `<p class="hint">The first day is still running. History fills in at midnight.</p>`;
      $("daily").innerHTML = "";
      $("daily").closest(".block").hidden = true;
      return;
    }
    $("daily").closest(".block").hidden = false;
    const W = 1000, H = 290, L0 = 44, R0 = 44, T0 = 14, B0 = 40;
    const maxSoil = Math.max(100, ...rows.map((r) => r.soil_water_mm_at_decision || 0));
    const maxRain = Math.max(10, ...rows.map((r) => r.rain_mm));
    const x = (i) => L0 + (rows.length === 1 ? 0.5 : i / (rows.length - 1)) * (W - L0 - R0);
    const y = (v) => T0 + (1 - v / maxSoil) * (H - T0 - B0);
    const bw = Math.min(26, Math.max(2, (W - L0 - R0) / rows.length * 0.6));
    const rain = rows.map((r, i) => { const h = (r.rain_mm / maxRain) * (H - T0 - B0) * 0.45; return `<rect x="${x(i) - bw / 2}" y="${H - B0 - h}" width="${bw}" height="${h}" fill="#9fbcd1"/>`; }).join("");
    const pts = rows.map((r, i) => (r.soil_water_mm_at_decision == null ? null : `${x(i)},${y(r.soil_water_mm_at_decision)}`)).filter(Boolean).join(" ");
    const irr = rows.map((r, i) => (r.delivered_l > 0 ? `<path d="M${x(i) - 6} ${H - B0 + 16} h12 l-6 -9z" fill="#2e7d32"><title>${esc(r.date)}: watered ${num(r.delivered_l, 2)} L</title></path>` : "")).join("");
    const step = Math.ceil(rows.length / 10);
    const ticks = rows.map((r, i) => (i % step === 0 ? `<text x="${x(i)}" y="${H - 4}" text-anchor="middle">${esc(dayLabel(r.date))}</text>` : "")).join("");
    $("chart").innerHTML = `<svg viewBox="0 0 ${W} ${H}" role="img" aria-label="Soil water at each morning decision, rain and watering days">
      ${rain}
      <line x1="${L0}" x2="${W - R0}" y1="${y(thr)}" y2="${y(thr)}" stroke="#c62828" stroke-dasharray="5 4"/>
      <polyline points="${pts}" fill="none" stroke="#1f5f8b" stroke-width="2.5"/>
      ${irr}
      <g class="axis"><text x="4" y="${y(maxSoil) + 10}">${maxSoil.toFixed(0)} mm</text><text x="4" y="${y(thr) + 4}">${thr.toFixed(0)} mm</text>
${ticks}</g></svg>
      <div class="legend"><span><i style="background:#1f5f8b"></i>Soil water at 05:00</span><span><i style="background:#c62828"></i>Stress line</span>
      <span><i style="background:#9fbcd1;height:10px"></i>Rain (tallest bar ${maxRain.toFixed(0)} mm)</span><span><i style="background:#2e7d32;height:10px"></i>Watering day</span></div>`;
    $("daily").innerHTML = `<thead><tr><th>Date</th><th>Decision</th><th class="n">Soil water</th><th class="n">Rain</th><th class="n">Water use</th><th class="n">Watered</th><th class="n">Nutrient EC</th><th class="n">Salt recovered</th></tr></thead><tbody>` +
      rows.slice().reverse().map((r) => `<tr><td>${esc(dayLabel(r.date))}</td><td>${esc(DECISION_LABEL[r.decision] || "No decision")}</td>
        <td class="n">${num(r.soil_water_mm_at_decision, 0)} mm</td><td class="n">${num(r.rain_mm)} mm</td>
        <td class="n">${num(r.et0_actual_mm)} mm</td><td class="n">${r.delivered_l > 0 ? `${num(r.delivered_l, 2)} L` : "–"}</td>
        <td class="n">${r.applied_ec_setpoint_dS_m ? num(r.applied_ec_setpoint_dS_m, 2) : "–"}</td><td class="n">${num(r.salt_recovered_g, 1)} g</td></tr>`).join("") + "</tbody>";
  }

  // ---------- alerts ----------
  function renderAlerts() {
    const list = events.filter((e) => alertFilter === "all" || e.level === "warn" || e.level === "alarm").slice().reverse();
    $("events").innerHTML = list.length
      ? list.map((e) => `<li class="lvl-${e.level}"><time>${esc(dayLabel(e.t))} ${esc(hhmm(e.t))}</time><span class="src">${esc(e.source.replace("esp32_", "ESP32 ").replace("pi5", "Raspberry Pi"))}</span><span>${friendly(e.msg)}</span></li>`).join("")
      : `<li><span class="empty">${alertFilter === "all" ? "Nothing logged yet." : "Nothing needs attention."}</span></li>`;
  }
  document.querySelectorAll(".chip").forEach((c) => c.addEventListener("click", () => {
    alertFilter = c.dataset.level;
    document.querySelectorAll(".chip").forEach((x) => x.classList.toggle("on", x === c));
    renderAlerts();
  }));

  // ---------- start ----------
  showView(location.hash.slice(1) || "today");
  refresh();
  setInterval(refresh, 1000);
})();
