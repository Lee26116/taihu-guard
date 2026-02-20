/**
 * TaihuGuard — ECharts 图表模块
 * V2: 时间序列 + 特征重要性 + 水质饼图 + 雷达图 + 站点对比
 */

let timeSeriesChart = null;
let featureImportanceChart = null;
let modalChart = null;
let wqDistChart = null;
let radarChart = null;
let stationCompareChart = null;

function initCharts() {
    const initOne = (id) => {
        const el = document.getElementById(id);
        if (!el) return null;
        const chart = echarts.init(el, null, { renderer: 'canvas' });
        return chart;
    };

    timeSeriesChart = initOne('timeSeriesChart');
    featureImportanceChart = initOne('featureImportanceChart');
    wqDistChart = initOne('wqDistChart');
    radarChart = initOne('radarChart');
    stationCompareChart = initOne('stationCompareChart');

    // 统一 resize
    let resizeTimer;
    window.addEventListener('resize', () => {
        clearTimeout(resizeTimer);
        resizeTimer = setTimeout(() => {
            [timeSeriesChart, featureImportanceChart, wqDistChart, radarChart, stationCompareChart, modalChart]
                .forEach(c => c?.resize());
        }, 150);
    });

    // 对比参数切换
    document.getElementById('compareParamSelect')?.addEventListener('change', (e) => {
        updateStationCompareChart(e.target.value);
    });
}

// 辅助: 判断窄屏
function isMobile() { return window.innerWidth <= 768; }

// ==================== 时间序列图 ====================
function updateTimeSeriesChart() {
    if (!timeSeriesChart) return;

    const stationId = AppState.selectedStation;
    const param = AppState.selectedParam;
    const station = AppState.stations.find(s => s.id === stationId);

    if (!station) {
        timeSeriesChart.clear();
        return;
    }

    const cfg = PARAM_CONFIG[param];

    const now = new Date();
    const historyDates = [];
    const historyValues = [];
    const currentVal = station.current?.[param] || 0;

    for (let i = 7; i >= 0; i--) {
        const d = new Date(now);
        d.setDate(d.getDate() - i);
        historyDates.push(formatDate(d));
        if (i === 0) {
            historyValues.push(currentVal);
        } else {
            const noise = (Math.random() - 0.5) * currentVal * 0.2;
            historyValues.push(Math.max(0, currentVal + noise));
        }
    }

    const predDates = [];
    const predValues = [];
    const predUpper = [];
    const predLower = [];

    if (station.predictions) {
        station.predictions.forEach(pred => {
            predDates.push(pred.date);
            const val = pred.values?.[param] || 0;
            predValues.push(val);
            const unc = pred.uncertainty?.[param] || val * 0.15;
            predUpper.push(val + unc);
            predLower.push(Math.max(0, val - unc));
        });
    }

    const allDates = [...historyDates, ...predDates];
    const historyFull = [...historyValues, ...Array(predDates.length).fill(null)];
    const predFull = [...Array(historyDates.length - 1).fill(null), currentVal, ...predValues];
    // 置信区间起点=当前值（无不确定性），逐渐展开
    const upperFull = [...Array(historyDates.length - 1).fill(null), currentVal, ...predUpper];
    const lowerFull = [...Array(historyDates.length - 1).fill(null), currentVal, ...predLower];

    const forecastStartIdx = historyDates.length - 1;

    const option = {
        backgroundColor: 'transparent',
        tooltip: {
            trigger: 'axis',
            backgroundColor: 'rgba(26, 31, 46, 0.95)',
            borderColor: '#2a3142',
            borderWidth: 1,
            textStyle: { color: '#e5e7eb', fontSize: 12 },
            formatter: function (params) {
                const dateStr = params[0].axisValue;
                const isForecast = params[0].dataIndex >= forecastStartIdx;
                let html = `<div style="font-weight:600;margin-bottom:4px">${dateStr} ${isForecast ? '<span style="color:#f97316;font-size:11px">[预测]</span>' : ''}</div>`;
                params.forEach(p => {
                    if (p.value !== null && p.value !== undefined && p.seriesName !== '置信下界' && p.seriesName !== '置信区间') {
                        html += `<div style="display:flex;align-items:center;gap:6px">
                            ${p.marker}
                            <span>${p.seriesName}: ${Number(p.value).toFixed(cfg.decimals)} ${cfg.unit}</span>
                        </div>`;
                    }
                });
                return html;
            }
        },
        legend: {
            data: ['实测值', '预测值'],
            top: 4,
            right: 16,
            textStyle: { color: '#9ca3af', fontSize: 11 },
            itemWidth: 16,
            itemHeight: 3
        },
        grid: { top: 40, right: isMobile() ? 12 : 24, bottom: 32, left: isMobile() ? 44 : 60 },
        xAxis: {
            type: 'category',
            data: allDates,
            axisLine: { lineStyle: { color: '#2a3142' } },
            axisLabel: { color: '#6b7280', fontSize: 11 },
            splitLine: { show: false }
        },
        yAxis: {
            type: 'value',
            name: `${cfg.name} (${cfg.unit})`,
            nameTextStyle: { color: '#6b7280', fontSize: 11 },
            axisLine: { show: false },
            axisLabel: { color: '#6b7280', fontSize: 11 },
            splitLine: { lineStyle: { color: '#1e2433', type: 'dashed' } }
        },
        series: [
            // 置信区间: 先画下界(透明), 再叠差值(填充)
            {
                name: '置信下界', type: 'line', data: lowerFull,
                lineStyle: { opacity: 0 }, stack: 'confidence', symbol: 'none', silent: true
            },
            {
                name: '置信区间', type: 'line',
                data: lowerFull.map((v, i) => {
                    if (v === null || upperFull[i] === null) return null;
                    return upperFull[i] - v;
                }),
                lineStyle: { opacity: 0 },
                areaStyle: {
                    color: new echarts.graphic.LinearGradient(0, 0, 0, 1, [
                        { offset: 0, color: 'rgba(0, 212, 255, 0.12)' },
                        { offset: 1, color: 'rgba(0, 212, 255, 0.02)' }
                    ])
                },
                stack: 'confidence', symbol: 'none', silent: true
            },
            {
                name: '实测值', type: 'line', data: historyFull, smooth: true,
                lineStyle: { color: '#00d4ff', width: 2 },
                itemStyle: { color: '#00d4ff' },
                areaStyle: {
                    color: new echarts.graphic.LinearGradient(0, 0, 0, 1, [
                        { offset: 0, color: 'rgba(0, 212, 255, 0.15)' },
                        { offset: 1, color: 'rgba(0, 212, 255, 0)' }
                    ])
                },
                symbolSize: 4, emphasis: { scale: true }
            },
            {
                name: '预测值', type: 'line', data: predFull, smooth: true,
                lineStyle: { color: '#f97316', width: 2, type: 'dashed' },
                itemStyle: { color: '#f97316' },
                areaStyle: {
                    color: new echarts.graphic.LinearGradient(0, 0, 0, 1, [
                        { offset: 0, color: 'rgba(249, 115, 22, 0.10)' },
                        { offset: 1, color: 'rgba(249, 115, 22, 0)' }
                    ])
                },
                symbolSize: 4, emphasis: { scale: true }
            },
            {
                name: '当前', type: 'line',
                markLine: {
                    silent: true, symbol: ['none', 'none'],
                    data: [{ xAxis: historyDates[historyDates.length - 1] }],
                    lineStyle: { color: '#6b7280', type: 'dashed', width: 1 },
                    label: { formatter: '当前', color: '#6b7280', fontSize: 10 }
                },
                markArea: {
                    silent: true,
                    data: predDates.length ? [[
                        { xAxis: historyDates[historyDates.length - 1] },
                        { xAxis: allDates[allDates.length - 1] }
                    ]] : [],
                    itemStyle: { color: 'rgba(249, 115, 22, 0.03)' },
                    label: { show: true, formatter: '预测区间', position: 'insideTop', color: 'rgba(249, 115, 22, 0.3)', fontSize: 10 }
                },
                data: []
            }
        ]
    };

    timeSeriesChart.setOption(option, true);

    // 联动更新雷达图
    updateRadarChart(station);
}

// ==================== 水质等级分布饼图 ====================
function updateWqDistChart() {
    if (!wqDistChart || !AppState.stations.length) return;

    const levelNames = ['I类', 'II类', 'III类', 'IV类', 'V类', '劣V类'];
    const levelColors = ['#22c55e', '#84cc16', '#eab308', '#f97316', '#ef4444', '#991b1b'];

    const counts = [0, 0, 0, 0, 0, 0];
    AppState.stations.forEach(s => {
        const lvl = (s.water_quality_level?.level || 3) - 1;
        if (lvl >= 0 && lvl < 6) counts[lvl]++;
    });

    const data = levelNames.map((name, i) => ({
        name, value: counts[i],
        itemStyle: { color: levelColors[i] }
    })).filter(d => d.value > 0);

    const option = {
        backgroundColor: 'transparent',
        tooltip: {
            trigger: 'item',
            backgroundColor: 'rgba(26, 31, 46, 0.95)',
            borderColor: '#2a3142',
            textStyle: { color: '#e5e7eb', fontSize: 12 },
            formatter: '{b}: {c}站 ({d}%)'
        },
        series: [{
            type: 'pie',
            radius: ['40%', '70%'],
            center: ['50%', '55%'],
            avoidLabelOverlap: true,
            itemStyle: { borderRadius: 4, borderColor: '#0a0e17', borderWidth: 2 },
            label: {
                color: '#9ca3af',
                fontSize: 11,
                formatter: '{b}\n{c}站'
            },
            labelLine: { lineStyle: { color: '#2a3142' } },
            emphasis: {
                label: { fontSize: 12, fontWeight: 600 },
                itemStyle: { shadowBlur: 10, shadowColor: 'rgba(0,0,0,0.3)' }
            },
            data: data
        }]
    };

    wqDistChart.setOption(option, true);
}

// ==================== 站点参数雷达图 ====================
function updateRadarChart(station) {
    if (!radarChart) return;

    if (!station) {
        station = AppState.stations.find(s => s.id === AppState.selectedStation);
    }
    if (!station) {
        radarChart.clear();
        return;
    }

    const params = ['chla', 'do', 'tp', 'tn', 'nh3n', 'water_temp'];
    const labels = ['Chl-a', 'DO', 'TP', 'TN', 'NH3-N', '水温'];

    // 各参数参考最大值（用于归一化）
    const maxRefs = { chla: 80, do: 14, tp: 0.3, tn: 5, nh3n: 2, water_temp: 35 };

    const values = params.map(p => {
        const val = station.current?.[p] || 0;
        const maxRef = maxRefs[p] || 1;
        return Math.min((val / maxRef) * 100, 100);
    });

    const option = {
        backgroundColor: 'transparent',
        tooltip: {
            backgroundColor: 'rgba(26, 31, 46, 0.95)',
            borderColor: '#2a3142',
            textStyle: { color: '#e5e7eb', fontSize: 12 }
        },
        radar: {
            indicator: labels.map(name => ({ name, max: 100 })),
            center: ['50%', '55%'],
            radius: '60%',
            shape: 'polygon',
            splitNumber: 4,
            axisName: { color: '#9ca3af', fontSize: 10 },
            splitLine: { lineStyle: { color: '#1e2433' } },
            splitArea: { areaStyle: { color: ['rgba(0,212,255,0.02)', 'transparent'] } },
            axisLine: { lineStyle: { color: '#2a3142' } }
        },
        series: [{
            type: 'radar',
            data: [{
                value: values,
                name: station.name,
                areaStyle: {
                    color: new echarts.graphic.LinearGradient(0, 0, 0, 1, [
                        { offset: 0, color: 'rgba(0, 212, 255, 0.3)' },
                        { offset: 1, color: 'rgba(0, 212, 255, 0.05)' }
                    ])
                },
                lineStyle: { color: '#00d4ff', width: 2 },
                itemStyle: { color: '#00d4ff' },
                symbolSize: 4
            }],
            tooltip: {
                trigger: 'item',
                formatter: function () {
                    let html = `<div style="font-weight:600;margin-bottom:4px">${station.name}</div>`;
                    params.forEach((p, i) => {
                        const cfg = PARAM_CONFIG[p];
                        const val = station.current?.[p];
                        html += `<div>${cfg.name}: ${val != null ? Number(val).toFixed(cfg.decimals) : '--'} ${cfg.unit}</div>`;
                    });
                    return html;
                }
            }
        }]
    };

    radarChart.setOption(option, true);
}

// ==================== 全湖站点参数对比柱状图 ====================
function updateStationCompareChart(param) {
    if (!stationCompareChart) return;
    if (!param) param = 'chla';
    if (!AppState.stations.length) return;

    const cfg = PARAM_CONFIG[param];

    // 按值排序，取所有站点
    const sorted = AppState.stations
        .map(s => ({ name: s.name?.replace(/监测站|站$/, '') || s.id, val: s.current?.[param] || 0, level: s.water_quality_level?.level || 3 }))
        .sort((a, b) => b.val - a.val);

    const levelColors = { 1: '#22c55e', 2: '#84cc16', 3: '#eab308', 4: '#f97316', 5: '#ef4444', 6: '#991b1b' };

    const option = {
        backgroundColor: 'transparent',
        tooltip: {
            trigger: 'axis',
            axisPointer: { type: 'shadow' },
            backgroundColor: 'rgba(26, 31, 46, 0.95)',
            borderColor: '#2a3142',
            textStyle: { color: '#e5e7eb', fontSize: 12 },
            formatter: (ps) => {
                const p = ps[0];
                return `${p.name}<br/>${cfg.name}: <b>${Number(p.value).toFixed(cfg.decimals)}</b> ${cfg.unit}`;
            }
        },
        grid: { top: 8, right: 12, bottom: 36, left: 12, containLabel: true },
        xAxis: {
            type: 'category',
            data: sorted.map(s => s.name),
            axisLine: { lineStyle: { color: '#2a3142' } },
            axisLabel: { color: '#6b7280', fontSize: isMobile() ? 8 : 9, rotate: isMobile() ? 50 : 35, interval: 0 },
            axisTick: { show: false }
        },
        yAxis: {
            type: 'value',
            name: cfg.unit,
            nameTextStyle: { color: '#6b7280', fontSize: 10 },
            axisLine: { show: false },
            axisLabel: { color: '#6b7280', fontSize: 10 },
            splitLine: { lineStyle: { color: '#1e2433', type: 'dashed' } }
        },
        series: [{
            type: 'bar',
            data: sorted.map(s => ({
                value: s.val,
                itemStyle: {
                    color: new echarts.graphic.LinearGradient(0, 0, 0, 1, [
                        { offset: 0, color: levelColors[s.level] || '#eab308' },
                        { offset: 1, color: (levelColors[s.level] || '#eab308') + '40' }
                    ]),
                    borderRadius: [3, 3, 0, 0]
                }
            })),
            barMaxWidth: 24,
            emphasis: {
                itemStyle: { shadowBlur: 8, shadowColor: 'rgba(0,0,0,0.3)' }
            }
        }]
    };

    stationCompareChart.setOption(option, true);
}

// ==================== 特征重要性图 ====================
function updateFeatureImportanceChart() {
    if (!featureImportanceChart || !AppState.modelMetrics) return;

    const features = AppState.modelMetrics.feature_importance || [];
    if (!features.length) return;

    const sorted = [...features].sort((a, b) => a.importance - b.importance);
    const maxImportance = Math.max(...sorted.map(f => f.importance), 0.01);

    const option = {
        backgroundColor: 'transparent',
        tooltip: {
            trigger: 'axis',
            axisPointer: { type: 'shadow' },
            backgroundColor: 'rgba(26, 31, 46, 0.95)',
            borderColor: '#2a3142',
            borderWidth: 1,
            textStyle: { color: '#e5e7eb', fontSize: 12 }
        },
        grid: { top: 8, right: 16, bottom: 8, left: 80, containLabel: false },
        xAxis: {
            type: 'value',
            axisLine: { show: false },
            axisLabel: { color: '#6b7280', fontSize: 10 },
            splitLine: { lineStyle: { color: '#1e2433', type: 'dashed' } }
        },
        yAxis: {
            type: 'category',
            data: sorted.map(f => f.feature),
            axisLine: { show: false },
            axisTick: { show: false },
            axisLabel: { color: '#9ca3af', fontSize: 11 }
        },
        series: [{
            type: 'bar',
            data: sorted.map(f => ({
                value: f.importance,
                itemStyle: {
                    color: new echarts.graphic.LinearGradient(0, 0, 1, 0, [
                        { offset: 0, color: 'rgba(0, 212, 255, 0.4)' },
                        { offset: f.importance / maxImportance, color: '#00d4ff' }
                    ])
                }
            })),
            barWidth: 14,
            itemStyle: { borderRadius: [0, 4, 4, 0] },
            label: {
                show: true, position: 'right',
                formatter: (p) => typeof p.value === 'number' ? p.value.toFixed(3) : p.value,
                color: '#6b7280', fontSize: 10
            }
        }]
    };

    featureImportanceChart.setOption(option, true);
}

// ==================== 站点详情弹窗图表 ====================
function renderModalChart(station) {
    const container = document.getElementById('modalChart');
    if (!container) return;

    if (modalChart) modalChart.dispose();
    modalChart = echarts.init(container, null, { renderer: 'canvas' });

    if (!station.predictions || !station.predictions.length) {
        modalChart.clear();
        return;
    }

    const params = ['chla', 'do', 'tp', 'nh3n'];
    const colors = ['#00d4ff', '#22c55e', '#f97316', '#a855f7'];
    const dates = station.predictions.map(p => p.date);

    const series = params.map((param, i) => ({
        name: PARAM_CONFIG[param].name,
        type: 'line',
        data: station.predictions.map(p => p.values?.[param] || 0),
        smooth: true,
        lineStyle: { color: colors[i], width: 2 },
        itemStyle: { color: colors[i] },
        areaStyle: {
            color: new echarts.graphic.LinearGradient(0, 0, 0, 1, [
                { offset: 0, color: colors[i] + '18' },
                { offset: 1, color: colors[i] + '00' }
            ])
        },
        symbolSize: 4
    }));

    const option = {
        backgroundColor: 'transparent',
        title: { text: '未来14天预测趋势', left: 0, textStyle: { color: '#e5e7eb', fontSize: 13, fontWeight: 600 } },
        tooltip: {
            trigger: 'axis',
            backgroundColor: 'rgba(26, 31, 46, 0.95)',
            borderColor: '#2a3142',
            textStyle: { color: '#e5e7eb', fontSize: 12 }
        },
        legend: {
            data: params.map(p => PARAM_CONFIG[p].name),
            top: 0, right: 0,
            textStyle: { color: '#9ca3af', fontSize: 11 },
            itemWidth: 16, itemHeight: 3
        },
        grid: { top: 40, right: 16, bottom: 24, left: 48 },
        xAxis: { type: 'category', data: dates, axisLine: { lineStyle: { color: '#2a3142' } }, axisLabel: { color: '#6b7280', fontSize: 10 } },
        yAxis: { type: 'value', axisLine: { show: false }, axisLabel: { color: '#6b7280', fontSize: 10 }, splitLine: { lineStyle: { color: '#1e2433', type: 'dashed' } } },
        series: series
    };

    modalChart.setOption(option, true);
}

// ==================== 工具函数 ====================
function formatDate(date) {
    const m = String(date.getMonth() + 1).padStart(2, '0');
    const d = String(date.getDate()).padStart(2, '0');
    return `${m}-${d}`;
}
