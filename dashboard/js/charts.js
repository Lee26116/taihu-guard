/**
 * TaihuGuard — ECharts 图表模块
 * 时间序列图、特征重要性图、站点详情图
 */

let timeSeriesChart = null;
let featureImportanceChart = null;
let modalChart = null;

function initCharts() {
    // 时间序列图
    const tsContainer = document.getElementById('timeSeriesChart');
    if (tsContainer) {
        timeSeriesChart = echarts.init(tsContainer, null, { renderer: 'canvas' });
        window.addEventListener('resize', () => timeSeriesChart?.resize());
    }

    // 特征重要性图
    const fiContainer = document.getElementById('featureImportanceChart');
    if (fiContainer) {
        featureImportanceChart = echarts.init(fiContainer, null, { renderer: 'canvas' });
        window.addEventListener('resize', () => featureImportanceChart?.resize());
    }
}

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

    // 历史数据 (模拟 — 实际从 API 获取)
    const now = new Date();
    const historyDates = [];
    const historyValues = [];
    const currentVal = station.current?.[param] || 0;

    // 生成 7 天历史 (简化: 基于当前值加噪声)
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

    // 预测数据
    const predDates = [];
    const predValues = [];
    const predUpper = [];
    const predLower = [];

    if (station.predictions) {
        station.predictions.forEach(pred => {
            predDates.push(pred.date);
            const val = pred.values?.[param] || 0;
            predValues.push(val);
            // 置信区间 (±15%)
            predUpper.push(val * 1.15);
            predLower.push(val * 0.85);
        });
    }

    // 合并日期轴
    const allDates = [...historyDates, ...predDates];
    const historyFull = [...historyValues, ...Array(predDates.length).fill(null)];
    const predFull = [...Array(historyDates.length - 1).fill(null), currentVal, ...predValues];
    const upperFull = [...Array(historyDates.length - 1).fill(null), currentVal * 1.15, ...predUpper];
    const lowerFull = [...Array(historyDates.length - 1).fill(null), currentVal * 0.85, ...predLower];

    const option = {
        backgroundColor: 'transparent',
        tooltip: {
            trigger: 'axis',
            backgroundColor: '#1a1f2e',
            borderColor: '#2a3142',
            textStyle: { color: '#e5e7eb', fontSize: 12 },
            formatter: function (params) {
                let html = `<div style="font-weight:600;margin-bottom:4px">${params[0].axisValue}</div>`;
                params.forEach(p => {
                    if (p.value !== null && p.value !== undefined) {
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
        grid: {
            top: 40,
            right: 24,
            bottom: 32,
            left: 60
        },
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
            // 置信区间上界（隐形线）
            {
                name: '置信上界',
                type: 'line',
                data: upperFull,
                lineStyle: { opacity: 0 },
                stack: 'confidence',
                symbol: 'none',
                silent: true
            },
            // 置信区间填充
            {
                name: '置信区间',
                type: 'line',
                data: lowerFull.map((v, i) => {
                    if (v === null || upperFull[i] === null) return null;
                    return upperFull[i] - v;
                }),
                lineStyle: { opacity: 0 },
                areaStyle: { color: 'rgba(0, 212, 255, 0.08)' },
                stack: 'confidence',
                symbol: 'none',
                silent: true
            },
            // 历史实测值
            {
                name: '实测值',
                type: 'line',
                data: historyFull,
                smooth: true,
                lineStyle: { color: '#00d4ff', width: 2 },
                itemStyle: { color: '#00d4ff' },
                symbolSize: 4,
                emphasis: { scale: true }
            },
            // 预测值
            {
                name: '预测值',
                type: 'line',
                data: predFull,
                smooth: true,
                lineStyle: { color: '#f97316', width: 2, type: 'dashed' },
                itemStyle: { color: '#f97316' },
                symbolSize: 4,
                emphasis: { scale: true }
            },
            // 当前时刻标记线
            {
                name: '当前',
                type: 'line',
                markLine: {
                    silent: true,
                    symbol: ['none', 'none'],
                    data: [{ xAxis: historyDates[historyDates.length - 1] }],
                    lineStyle: { color: '#6b7280', type: 'dashed', width: 1 },
                    label: {
                        formatter: '当前',
                        color: '#6b7280',
                        fontSize: 10
                    }
                },
                data: []
            }
        ]
    };

    timeSeriesChart.setOption(option, true);
}

// ==================== 特征重要性图 ====================
function updateFeatureImportanceChart() {
    if (!featureImportanceChart || !AppState.modelMetrics) return;

    const features = AppState.modelMetrics.feature_importance || [];
    if (!features.length) return;

    const sorted = [...features].sort((a, b) => a.importance - b.importance);

    const option = {
        backgroundColor: 'transparent',
        title: {
            text: 'SHAP Feature Importance',
            left: 0,
            top: 0,
            textStyle: { color: '#e5e7eb', fontSize: 13, fontWeight: 600 }
        },
        tooltip: {
            trigger: 'axis',
            axisPointer: { type: 'shadow' },
            backgroundColor: '#1a1f2e',
            borderColor: '#2a3142',
            textStyle: { color: '#e5e7eb', fontSize: 12 }
        },
        grid: {
            top: 36,
            right: 16,
            bottom: 8,
            left: 80,
            containLabel: false
        },
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
            data: sorted.map(f => f.importance),
            barWidth: 14,
            itemStyle: {
                color: new echarts.graphic.LinearGradient(0, 0, 1, 0, [
                    { offset: 0, color: '#00d4ff' },
                    { offset: 1, color: '#0099cc' }
                ]),
                borderRadius: [0, 4, 4, 0]
            },
            label: {
                show: true,
                position: 'right',
                formatter: '{c}',
                color: '#6b7280',
                fontSize: 10
            }
        }]
    };

    featureImportanceChart.setOption(option, true);
}

// ==================== 站点详情弹窗图表 ====================
function renderModalChart(station) {
    const container = document.getElementById('modalChart');
    if (!container) return;

    if (modalChart) {
        modalChart.dispose();
    }
    modalChart = echarts.init(container, null, { renderer: 'canvas' });

    if (!station.predictions || !station.predictions.length) {
        modalChart.clear();
        return;
    }

    // 展示多参数预测趋势
    const params = ['chla', 'do', 'tp', 'nh3n'];
    const colors = ['#00d4ff', '#22c55e', '#f97316', '#a855f7'];

    const dates = station.predictions.map(p => p.date);

    const series = params.map((param, i) => {
        const cfg = PARAM_CONFIG[param];
        return {
            name: cfg.name,
            type: 'line',
            data: station.predictions.map(p => p.values?.[param] || 0),
            smooth: true,
            lineStyle: { color: colors[i], width: 2 },
            itemStyle: { color: colors[i] },
            symbolSize: 4
        };
    });

    const option = {
        backgroundColor: 'transparent',
        title: {
            text: '未来7天预测趋势',
            left: 0,
            textStyle: { color: '#e5e7eb', fontSize: 13, fontWeight: 600 }
        },
        tooltip: {
            trigger: 'axis',
            backgroundColor: '#1a1f2e',
            borderColor: '#2a3142',
            textStyle: { color: '#e5e7eb', fontSize: 12 }
        },
        legend: {
            data: params.map(p => PARAM_CONFIG[p].name),
            top: 0,
            right: 0,
            textStyle: { color: '#9ca3af', fontSize: 11 },
            itemWidth: 16,
            itemHeight: 3
        },
        grid: {
            top: 40,
            right: 16,
            bottom: 24,
            left: 48
        },
        xAxis: {
            type: 'category',
            data: dates,
            axisLine: { lineStyle: { color: '#2a3142' } },
            axisLabel: { color: '#6b7280', fontSize: 10 }
        },
        yAxis: {
            type: 'value',
            axisLine: { show: false },
            axisLabel: { color: '#6b7280', fontSize: 10 },
            splitLine: { lineStyle: { color: '#1e2433', type: 'dashed' } }
        },
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
