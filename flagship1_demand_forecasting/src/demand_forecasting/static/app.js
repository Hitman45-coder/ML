const form = document.querySelector('#forecast-form');
const errorBox = document.querySelector('#form-error');
const button = document.querySelector('#submit-button');
const healthPill = document.querySelector('#health-pill');

function setHealth(ok, text) {
  healthPill.textContent = text;
  healthPill.className = `pill ${ok ? 'pill-ok' : 'pill-bad'}`;
}

async function loadHealth() {
  try {
    const response = await fetch('/health');
    const payload = await response.json();
    setHealth(payload.model_loaded, payload.model_loaded ? `Model ${payload.model_version} ready` : 'Model unavailable');
  } catch (_) { setHealth(false, 'API unavailable'); }
}

async function loadMetrics() {
  const grid = document.querySelector('#metrics-grid');
  try {
    const response = await fetch('/v1/metrics');
    if (!response.ok) throw new Error('metrics unavailable');
    const data = await response.json();
    const values = [
      ['Test MAE', data.test_model.mae.toFixed(2), 'bikes'],
      ['Test RMSE', data.test_model.rmse.toFixed(2), 'bikes'],
      ['Test RMSLE', data.test_model.rmsle.toFixed(3), 'log scale'],
    ];
    grid.innerHTML = values.map(([name, value, suffix]) => `<div class="metric"><span>${name}</span><strong>${value}</strong><small>${suffix}</small></div>`).join('');
    document.querySelector('#test-mae').textContent = `${data.test_model.mae.toFixed(1)} MAE`;
  } catch (_) { grid.innerHTML = '<div class="metric"><span>Metrics report unavailable</span><small>Run the training command first.</small></div>'; }
}

form.addEventListener('submit', async (event) => {
  event.preventDefault(); errorBox.textContent = ''; button.disabled = true; button.firstChild.textContent = 'Generating… ';
  const values = Object.fromEntries(new FormData(form).entries());
  const body = {
    // Keep the hour selected by the operator. The API treats this historical dataset timestamp as local/naive;
    // converting it to UTC would shift the hour for users outside UTC.
    timestamp: `${values.timestamp}:00`, season: Number(values.season), weathersit: Number(values.weathersit),
    temperature_c: Number(values.temperature_c), feels_like_c: Number(values.feels_like_c),
    humidity_percent: Number(values.humidity_percent), windspeed_kmh: Number(values.windspeed_kmh),
    workingday: form.workingday.checked ? 1 : 0, holiday: form.holiday.checked ? 1 : 0,
  };
  try {
    const response = await fetch('/v1/forecast', { method: 'POST', headers: {'content-type': 'application/json'}, body: JSON.stringify(body) });
    const payload = await response.json();
    if (!response.ok) {
      const detail = Array.isArray(payload.detail)
        ? payload.detail.map((item) => item.msg).join('; ')
        : payload.detail;
      throw new Error(detail || 'Forecast request failed');
    }
    document.querySelector('#result-empty').classList.add('hidden'); document.querySelector('#result-content').classList.remove('hidden');
    document.querySelector('#forecast-value').textContent = Math.round(payload.forecast_demand).toLocaleString();
    const rec = document.querySelector('#recommendation'); rec.textContent = payload.recommendation; rec.className = `recommendation ${payload.recommendation}`;
    document.querySelector('#band-value').textContent = `${Math.round(payload.planning_lower).toLocaleString()} – ${Math.round(payload.planning_upper).toLocaleString()}`;
    document.querySelector('#band-fill').style.width = `${Math.min(100, Math.max(18, payload.planning_upper / Math.max(1, payload.forecast_demand * 2) * 100))}%`;
    document.querySelector('#model-version').textContent = payload.model_version; document.querySelector('#latency').textContent = `${payload.latency_ms} ms`; document.querySelector('#result-time').textContent = new Date().toLocaleTimeString();
  } catch (error) { errorBox.textContent = error.message; }
  finally { button.disabled = false; button.firstChild.textContent = 'Generate forecast '; }
});

loadHealth(); loadMetrics();
