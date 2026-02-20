/**
 * TaihuGuard — Mapbox 地图模块
 * 太湖流域地图可视化: 站点标记、热力图、水质等级着色
 */

let map = null;
let mapMarkers = [];
let mapPopup = null;

function initMap() {
    mapboxgl.accessToken = CONFIG.MAPBOX_TOKEN;

    map = new mapboxgl.Map({
        container: 'map',
        style: 'mapbox://styles/mapbox/dark-v11',
        center: CONFIG.TAIHU_CENTER,
        zoom: CONFIG.TAIHU_ZOOM,
        pitch: 0,
        bearing: 0,
        antialias: true
    });

    // 地图控件
    map.addControl(new mapboxgl.NavigationControl(), 'top-left');
    map.addControl(new mapboxgl.ScaleControl({ unit: 'metric' }), 'bottom-right');

    map.on('load', () => {
        // 添加太湖轮廓（简化）
        addTaihuOutline();

        // 添加站点标记
        updateMapMarkers();

        // 添加热力图层
        addHeatmapLayer();
    });
}

function addTaihuOutline() {
    // 太湖简化轮廓 GeoJSON
    const taihuOutline = {
        type: 'Feature',
        geometry: {
            type: 'Polygon',
            coordinates: [[
                [119.92, 31.40], [120.00, 31.42], [120.10, 31.42],
                [120.20, 31.40], [120.30, 31.38], [120.38, 31.32],
                [120.42, 31.25], [120.48, 31.18], [120.50, 31.10],
                [120.48, 31.02], [120.42, 30.95], [120.35, 30.92],
                [120.25, 30.90], [120.15, 30.92], [120.05, 30.95],
                [119.95, 31.00], [119.90, 31.08], [119.88, 31.18],
                [119.90, 31.28], [119.92, 31.35], [119.92, 31.40]
            ]]
        }
    };

    map.addSource('taihu-outline', {
        type: 'geojson',
        data: taihuOutline
    });

    map.addLayer({
        id: 'taihu-fill',
        type: 'fill',
        source: 'taihu-outline',
        paint: {
            'fill-color': '#00d4ff',
            'fill-opacity': 0.05
        }
    });

    map.addLayer({
        id: 'taihu-border',
        type: 'line',
        source: 'taihu-outline',
        paint: {
            'line-color': '#00d4ff',
            'line-width': 1.5,
            'line-opacity': 0.4,
            'line-dasharray': [4, 2]
        }
    });
}

function updateMapMarkers() {
    // 清除现有标记
    mapMarkers.forEach(m => m.remove());
    mapMarkers = [];

    if (!AppState.stations.length) return;

    // 准备 GeoJSON 数据源
    const geojson = {
        type: 'FeatureCollection',
        features: AppState.stations.map(station => ({
            type: 'Feature',
            geometry: {
                type: 'Point',
                coordinates: [station.lon, station.lat]
            },
            properties: {
                id: station.id,
                name: station.name,
                basin: station.basin || '',
                type: station.type || '',
                level: station.water_quality_level?.level || 3,
                levelName: station.water_quality_level?.name || 'III类',
                levelColor: station.water_quality_level?.color || '#eab308',
                bloomLevel: station.bloom_warning?.level || 0,
                bloomLabel: station.bloom_warning?.label || '无风险',
                bloomColor: station.bloom_warning?.color || '#22c55e',
                ...station.current
            }
        }))
    };

    // 为每个站点创建自定义标记
    geojson.features.forEach(feature => {
        const props = feature.properties;
        const [lng, lat] = feature.geometry.coordinates;

        // 创建标记 DOM
        const el = document.createElement('div');
        el.className = 'station-marker';
        el.style.cssText = `
            width: 16px;
            height: 16px;
            border-radius: 50%;
            background: ${props.levelColor};
            border: 2px solid rgba(255,255,255,0.8);
            cursor: pointer;
            transition: transform 0.2s;
            box-shadow: 0 0 8px ${props.levelColor}80;
        `;

        // 蓝藻预警站点加大
        if (props.bloomLevel >= 2) {
            el.style.width = '22px';
            el.style.height = '22px';
            el.style.border = `3px solid ${props.bloomColor}`;
            el.style.animation = 'pulse 1.5s infinite';
        }

        el.addEventListener('mouseenter', () => {
            el.style.transform = 'scale(1.5)';
            showPopup(feature);
        });

        el.addEventListener('mouseleave', () => {
            el.style.transform = 'scale(1)';
        });

        el.addEventListener('click', () => {
            openStationModal(props.id);
        });

        const marker = new mapboxgl.Marker(el)
            .setLngLat([lng, lat])
            .addTo(map);

        mapMarkers.push(marker);
    });

    // 更新热力图数据
    if (map.getSource('heatmap-source')) {
        map.getSource('heatmap-source').setData(geojson);
    }
}

function showPopup(feature) {
    if (mapPopup) mapPopup.remove();

    const props = feature.properties;
    const [lng, lat] = feature.geometry.coordinates;

    const paramNames = {
        chla: 'Chl-a', do: 'DO', tp: 'TP', tn: 'TN',
        nh3n: 'NH3-N', codmn: 'CODMn', water_temp: '水温', ph: 'pH'
    };

    let paramsHtml = '<div class="popup-params">';
    ['chla', 'do', 'tp', 'tn', 'nh3n', 'codmn', 'water_temp', 'ph'].forEach(p => {
        const val = props[p];
        const cfg = PARAM_CONFIG[p];
        paramsHtml += `
            <div class="popup-param">
                <span class="name">${cfg.name}</span>
                <span class="val">${val !== null && val !== undefined ? Number(val).toFixed(cfg.decimals) : '--'} ${cfg.unit}</span>
            </div>
        `;
    });
    paramsHtml += '</div>';

    const html = `
        <div class="popup-title">
            ${props.name}
            <span class="popup-level" style="background:${props.levelColor}22;color:${props.levelColor}">${props.levelName}</span>
        </div>
        ${paramsHtml}
        <div class="popup-bloom">
            蓝藻预警: <span style="color:${props.bloomColor};font-weight:600">${props.bloomLabel}</span>
        </div>
    `;

    mapPopup = new mapboxgl.Popup({
        closeButton: true,
        closeOnClick: false,
        offset: 12,
        maxWidth: '280px'
    })
    .setLngLat([lng, lat])
    .setHTML(html)
    .addTo(map);
}

function addHeatmapLayer() {
    // 初始化空的热力图数据源
    const emptyGeojson = { type: 'FeatureCollection', features: [] };

    map.addSource('heatmap-source', {
        type: 'geojson',
        data: emptyGeojson
    });

    map.addLayer({
        id: 'heatmap-layer',
        type: 'heatmap',
        source: 'heatmap-source',
        paint: {
            // 权重基于叶绿素a浓度
            'heatmap-weight': [
                'interpolate', ['linear'],
                ['coalesce', ['get', 'chla'], 0],
                0, 0,
                10, 0.3,
                30, 0.6,
                60, 0.8,
                100, 1
            ],
            // 半径
            'heatmap-radius': [
                'interpolate', ['linear'], ['zoom'],
                8, 30,
                12, 60
            ],
            // 颜色梯度
            'heatmap-color': [
                'interpolate', ['linear'], ['heatmap-density'],
                0, 'rgba(0,0,0,0)',
                0.2, 'rgba(34,197,94,0.4)',
                0.4, 'rgba(132,204,22,0.5)',
                0.6, 'rgba(234,179,8,0.6)',
                0.8, 'rgba(249,115,22,0.7)',
                1, 'rgba(239,68,68,0.8)'
            ],
            // 透明度
            'heatmap-opacity': 0.6
        }
    }, 'taihu-fill'); // 放在太湖填充层下面
}

function updateMapHeatmap(param) {
    if (!map || !map.getLayer('heatmap-layer')) return;

    // 获取参数的合理范围用于插值
    const ranges = {
        chla: [0, 10, 30, 60, 100],
        do: [0, 4, 6, 8, 12],
        tp: [0, 0.05, 0.1, 0.2, 0.4],
        tn: [0, 1, 2, 4, 8],
        nh3n: [0, 0.3, 0.5, 1.0, 2.0],
        algae_density: [0, 500, 2000, 10000, 50000],
        water_temp: [0, 10, 20, 25, 35],
        ph: [5, 6.5, 7, 8, 9],
        codmn: [0, 2, 4, 6, 10]
    };

    const r = ranges[param] || [0, 25, 50, 75, 100];

    map.setPaintProperty('heatmap-layer', 'heatmap-weight', [
        'interpolate', ['linear'],
        ['coalesce', ['get', param], 0],
        r[0], 0,
        r[1], 0.3,
        r[2], 0.6,
        r[3], 0.8,
        r[4], 1
    ]);
}

// 飞行到指定站点
function flyToStation(lon, lat) {
    if (!map) return;
    map.flyTo({
        center: [lon, lat],
        zoom: 12,
        duration: 1500,
        essential: true
    });
}
