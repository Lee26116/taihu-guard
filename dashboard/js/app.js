/**
 * TaihuGuard — 主应用逻辑
 * 初始化、数据获取、全局状态管理
 */

// ==================== 配置 ====================
const CONFIG = {
    API_BASE: window.location.hostname === 'localhost' || window.location.hostname === '127.0.0.1'
        ? 'http://localhost:8087'
        : '',
    MAPBOX_TOKEN: '', // 从 /api/config 动态加载
    REFRESH_INTERVAL: 4 * 60 * 60 * 1000, // 4 小时
    TAIHU_CENTER: [120.1375, 31.2258],     // [lng, lat]
    TAIHU_ZOOM: 9.5,
};

// ==================== 全局状态 ====================
const AppState = {
    stations: [],
    alerts: [],
    latestData: null,
    selectedStation: null,
    selectedParam: 'chla',
    modelMetrics: null,
    isLoading: true,
};

// ==================== 参数配置 ====================
const PARAM_CONFIG = {
    water_temp: { name: '水温', unit: '°C', decimals: 1 },
    ph:         { name: 'pH', unit: '', decimals: 2 },
    do:         { name: '溶解氧', unit: 'mg/L', decimals: 2 },
    conductivity: { name: '电导率', unit: 'μS/cm', decimals: 0 },
    turbidity:  { name: '浊度', unit: 'NTU', decimals: 1 },
    codmn:      { name: 'CODMn', unit: 'mg/L', decimals: 2 },
    nh3n:       { name: '氨氮', unit: 'mg/L', decimals: 3 },
    tp:         { name: '总磷', unit: 'mg/L', decimals: 4 },
    tn:         { name: '总氮', unit: 'mg/L', decimals: 2 },
    chla:       { name: '叶绿素a', unit: 'μg/L', decimals: 1 },
    algae_density: { name: '藻密度', unit: '万cells/L', decimals: 0 },
};

// ==================== API 调用 ====================
async function fetchAPI(endpoint) {
    try {
        const resp = await fetch(`${CONFIG.API_BASE}${endpoint}`);
        if (!resp.ok) throw new Error(`HTTP ${resp.status}`);
        return await resp.json();
    } catch (err) {
        console.error(`API 请求失败 [${endpoint}]:`, err);
        return null;
    }
}

async function loadLatestData() {
    const data = await fetchAPI('/api/latest');
    if (data) {
        AppState.latestData = data;
        AppState.stations = data.stations || [];
        AppState.alerts = data.alerts || [];

        // 更新时间
        if (data.update_time) {
            const t = new Date(data.update_time);
            document.getElementById('updateTime').textContent =
                `更新: ${t.getFullYear()}-${String(t.getMonth()+1).padStart(2,'0')}-${String(t.getDate()).padStart(2,'0')} ${String(t.getHours()).padStart(2,'0')}:${String(t.getMinutes()).padStart(2,'0')}`;
        }

        // Demo 标识
        if (data.demo) {
            const statusText = document.querySelector('.status-text');
            if (statusText) statusText.textContent = 'Demo';
        }
    }
    return data;
}

async function loadModelMetrics() {
    const data = await fetchAPI('/api/model/metrics');
    if (data) {
        AppState.modelMetrics = data;
    }
    return data;
}

// ==================== UI 更新 ====================
function updateMetricCards() {
    if (!AppState.stations.length) return;

    // 计算所有站点的均值
    const params = ['chla', 'do', 'tp'];
    const ids = ['metricChla', 'metricDO', 'metricTP'];

    params.forEach((param, i) => {
        const values = AppState.stations
            .map(s => s.current?.[param])
            .filter(v => v !== null && v !== undefined);

        if (!values.length) return;

        const avg = values.reduce((a, b) => a + b, 0) / values.length;
        const config = PARAM_CONFIG[param];
        const card = document.getElementById(ids[i]);
        if (!card) return;

        card.querySelector('.metric-value').textContent =
            avg.toFixed(config.decimals);

        // 趋势: 对比预测值
        const futureValues = AppState.stations
            .map(s => s.predictions?.[0]?.values?.[param])
            .filter(v => v !== null && v !== undefined);

        if (futureValues.length) {
            const futureAvg = futureValues.reduce((a, b) => a + b, 0) / futureValues.length;
            const change = ((futureAvg - avg) / Math.max(Math.abs(avg), 0.001)) * 100;
            const trendEl = card.querySelector('.metric-trend');

            if (Math.abs(change) < 1) {
                trendEl.textContent = '→ 持平';
                trendEl.className = 'metric-trend stable';
            } else if (change > 0) {
                trendEl.textContent = `↑ ${change.toFixed(1)}%`;
                trendEl.className = 'metric-trend up';
            } else {
                trendEl.textContent = `↓ ${Math.abs(change).toFixed(1)}%`;
                trendEl.className = 'metric-trend down';
            }
        }
    });
}

function updateModelMetricsUI() {
    const m = AppState.modelMetrics;
    if (!m) return;

    const wq = m.water_quality || {};
    const bloom = m.bloom_warning || {};

    const setVal = (id, val) => {
        const el = document.getElementById(id);
        if (el && val !== undefined && val !== null) {
            el.textContent = typeof val === 'number' ? val.toFixed(val < 1 ? 4 : 2) : val;
        }
    };

    setVal('maeChla', wq.chla?.mae);
    setVal('r2Chla', wq.chla?.r2);
    setVal('f1Bloom', bloom.f1_macro);
    setVal('aucBloom', bloom.auc_roc);
    setVal('accBloom', bloom.accuracy);
}

function populateStationSelect() {
    const select = document.getElementById('chartStationSelect');
    if (!select || !AppState.stations.length) return;

    // 清除旧选项（保留第一个）
    while (select.options.length > 1) {
        select.remove(1);
    }

    AppState.stations.forEach(s => {
        const opt = document.createElement('option');
        opt.value = s.id;
        opt.textContent = `${s.name} (${s.basin || ''})`;
        select.appendChild(opt);
    });

    // 默认选第一个
    if (AppState.stations.length > 0) {
        select.value = AppState.stations[0].id;
        AppState.selectedStation = AppState.stations[0].id;
    }
}

// ==================== 事件绑定 ====================
function bindEvents() {
    // 参数 tab 切换
    document.getElementById('paramTabs')?.addEventListener('click', (e) => {
        if (!e.target.classList.contains('param-tab')) return;

        document.querySelectorAll('.param-tab').forEach(t => t.classList.remove('active'));
        e.target.classList.add('active');

        AppState.selectedParam = e.target.dataset.param;
        if (typeof updateTimeSeriesChart === 'function') {
            updateTimeSeriesChart();
        }
    });

    // 站点选择
    document.getElementById('chartStationSelect')?.addEventListener('change', (e) => {
        AppState.selectedStation = e.target.value;
        if (typeof updateTimeSeriesChart === 'function') {
            updateTimeSeriesChart();
        }
    });

    // 地图参数选择
    document.getElementById('mapParamSelect')?.addEventListener('change', (e) => {
        if (typeof updateMapHeatmap === 'function') {
            updateMapHeatmap(e.target.value);
        }
    });

    // 弹窗关闭
    document.getElementById('modalClose')?.addEventListener('click', closeStationModal);
    document.getElementById('stationModal')?.addEventListener('click', (e) => {
        if (e.target.id === 'stationModal') closeStationModal();
    });
}

function openStationModal(stationId) {
    const station = AppState.stations.find(s => s.id === stationId);
    if (!station) return;

    const modal = document.getElementById('stationModal');
    const nameEl = document.getElementById('modalStationName');
    const contentEl = document.getElementById('modalContent');

    nameEl.textContent = `${station.name} - ${station.basin || ''}`;

    // 构建详情内容
    const current = station.current || {};
    const level = station.water_quality_level || {};
    const bloom = station.bloom_warning || {};

    let html = `
        <div style="display:flex;align-items:center;gap:8px;margin-bottom:16px;">
            <span class="popup-level" style="background:${level.color}22;color:${level.color}">${level.name || '--'}</span>
            <span class="popup-level" style="background:${bloom.color}22;color:${bloom.color}">蓝藻: ${bloom.label || '--'}</span>
        </div>
        <div class="station-info-grid">
    `;

    const showParams = ['water_temp', 'ph', 'do', 'chla', 'tp', 'tn', 'nh3n', 'codmn', 'algae_density', 'turbidity', 'conductivity'];
    showParams.forEach(param => {
        const cfg = PARAM_CONFIG[param];
        const val = current[param];
        html += `
            <div class="station-info-item">
                <div class="label">${cfg.name}</div>
                <div class="value">${val !== null && val !== undefined ? Number(val).toFixed(cfg.decimals) : '--'}</div>
                <div class="unit">${cfg.unit}</div>
            </div>
        `;
    });

    html += '</div>';
    contentEl.innerHTML = html;

    // 弹窗图表
    if (typeof renderModalChart === 'function') {
        renderModalChart(station);
    }

    modal.classList.add('active');
}

function closeStationModal() {
    document.getElementById('stationModal')?.classList.remove('active');
}

// ==================== 初始化 ====================
async function init() {
    console.log('TaihuGuard 初始化...');

    bindEvents();

    // 加载配置 (Mapbox Token 等)
    const configData = await fetchAPI('/api/config');
    if (configData && configData.mapbox_token) {
        CONFIG.MAPBOX_TOKEN = configData.mapbox_token;
    }

    // 并行加载数据
    const [latestData, metrics] = await Promise.all([
        loadLatestData(),
        loadModelMetrics()
    ]);

    // 初始化地图
    if (typeof initMap === 'function') {
        initMap();
    }

    // 初始化图表
    if (typeof initCharts === 'function') {
        initCharts();
    }

    // 更新 UI
    populateStationSelect();
    updateMetricCards();
    updateModelMetricsUI();

    // 更新预警列表
    if (typeof updateAlertList === 'function') {
        updateAlertList();
    }

    // 更新图表
    if (typeof updateTimeSeriesChart === 'function') {
        updateTimeSeriesChart();
    }

    if (typeof updateFeatureImportanceChart === 'function') {
        updateFeatureImportanceChart();
    }

    AppState.isLoading = false;
    console.log('TaihuGuard 初始化完成');

    // 定时刷新
    setInterval(async () => {
        console.log('自动刷新数据...');
        await loadLatestData();
        updateMetricCards();
        if (typeof updateMapMarkers === 'function') updateMapMarkers();
        if (typeof updateAlertList === 'function') updateAlertList();
        if (typeof updateTimeSeriesChart === 'function') updateTimeSeriesChart();
    }, CONFIG.REFRESH_INTERVAL);
}

// 页面加载完成后初始化
document.addEventListener('DOMContentLoaded', init);
