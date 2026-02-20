/**
 * TaihuGuard — 预警通知模块
 * 管理蓝藻预警列表的显示和交互
 */

function updateAlertList() {
    const container = document.getElementById('alertList');
    const countBadge = document.getElementById('alertCount');
    if (!container) return;

    const alerts = AppState.alerts || [];

    // 更新计数
    if (countBadge) {
        countBadge.textContent = alerts.length;
    }

    // 没有预警
    if (!alerts.length) {
        // 如果也没有站点数据，显示加载中
        if (!AppState.stations.length) {
            container.innerHTML = '<div class="loading-placeholder">加载中...</div>';
            return;
        }

        // 计算各级预警站点
        const stationsWithWarning = AppState.stations.filter(s =>
            s.bloom_warning && s.bloom_warning.level >= 1
        );

        if (!stationsWithWarning.length) {
            container.innerHTML = `
                <div class="no-alerts">
                    <div style="font-size:24px;margin-bottom:8px">&#10003;</div>
                    全湖水质正常，暂无蓝藻预警
                </div>
            `;
            return;
        }

        // 从站点数据中提取预警
        const derived = stationsWithWarning
            .sort((a, b) => (b.bloom_warning?.level || 0) - (a.bloom_warning?.level || 0))
            .map(s => ({
                station_id: s.id,
                station_name: s.name,
                basin: s.basin || '',
                level: s.bloom_warning.level,
                label: s.bloom_warning.label,
                color: s.bloom_warning.color,
                lat: s.lat,
                lon: s.lon
            }));

        renderAlertItems(container, derived);
        if (countBadge) countBadge.textContent = derived.length;
        return;
    }

    renderAlertItems(container, alerts);
}

function renderAlertItems(container, alerts) {
    const levelClass = { 3: 'high', 2: 'mid', 1: 'low' };

    container.innerHTML = '';

    alerts.forEach(alert => {
        const cls = levelClass[alert.level] || 'low';

        const el = document.createElement('div');
        el.className = `alert-item alert-${cls}`;

        el.innerHTML = `
            <div class="alert-dot" style="background:${alert.color};box-shadow:0 0 6px ${alert.color}"></div>
            <div class="alert-info">
                <div class="alert-station">${escapeHtml(alert.station_name || '')}</div>
                <div class="alert-detail">${escapeHtml(alert.basin || '')}</div>
            </div>
            <span class="alert-level ${cls}">${escapeHtml(alert.label || '')}</span>
        `;

        el.addEventListener('click', () => {
            onAlertClick(alert.station_id, alert.lon, alert.lat);
        });

        container.appendChild(el);
    });
}

/** 防 XSS 转义 */
function escapeHtml(str) {
    const div = document.createElement('div');
    div.textContent = str;
    return div.innerHTML;
}

function onAlertClick(stationId, lon, lat) {
    // 飞行到站点
    if (typeof flyToStation === 'function' && lon != null && lat != null) {
        flyToStation(lon, lat);
    }

    // 更新图表
    AppState.selectedStation = stationId;
    const select = document.getElementById('chartStationSelect');
    if (select) select.value = stationId;

    if (typeof updateTimeSeriesChart === 'function') {
        updateTimeSeriesChart();
    }
}
