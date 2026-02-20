/**
 * TaihuGuard — 时间轴模块
 * 可拖动时间滑动条: -7天(历史) → 当前 → +14天(预测)
 * 拖动时更新地图站点颜色（按预测数据着色）
 */

const Timeline = {
    slider: null,
    dateLabel: null,
    progress: null,
    playBtn: null,
    playing: false,
    playTimer: null,
    // 0=最早(-7天), 7=当前, 21=最远(+14天)
    totalSteps: 21,
    currentStep: 7,

    init() {
        this.slider = document.getElementById('timelineSlider');
        this.dateLabel = document.getElementById('timelineDate');
        this.progress = document.getElementById('timelineProgress');
        this.playBtn = document.getElementById('tlPlay');
        const resetBtn = document.getElementById('tlReset');
        const ticksEl = document.getElementById('timelineTicks');

        if (!this.slider) return;

        // 生成刻度
        if (ticksEl) {
            let html = '';
            for (let i = 0; i <= this.totalSteps; i++) {
                const pct = (i / this.totalSteps) * 100;
                const isCurrent = i === 7;
                html += `<div class="tick${isCurrent ? ' tick-now' : ''}" style="left:${pct}%"></div>`;
            }
            ticksEl.innerHTML = html;
        }

        // 滑动事件
        this.slider.addEventListener('input', () => {
            this.currentStep = parseInt(this.slider.value);
            this.update();
        });

        // 播放按钮
        if (this.playBtn) {
            this.playBtn.addEventListener('click', () => this.togglePlay());
        }

        // 重置按钮
        if (resetBtn) {
            resetBtn.addEventListener('click', () => {
                this.stop();
                this.currentStep = 7;
                this.slider.value = 7;
                this.update();
            });
        }

        this.update();
    },

    update() {
        const dayOffset = this.currentStep - 7;
        const now = new Date();
        const targetDate = new Date(now.getTime() + dayOffset * 24 * 3600 * 1000);

        // 更新进度条
        if (this.progress) {
            const pct = (this.currentStep / this.totalSteps) * 100;
            this.progress.style.width = `${pct}%`;
        }

        // 更新日期标签
        if (this.dateLabel) {
            if (dayOffset === 0) {
                this.dateLabel.textContent = '当前';
            } else {
                const m = String(targetDate.getMonth() + 1).padStart(2, '0');
                const d = String(targetDate.getDate()).padStart(2, '0');
                const prefix = dayOffset > 0 ? '预测' : '历史';
                this.dateLabel.textContent = `${prefix} ${m}-${d} (${dayOffset > 0 ? '+' : ''}${dayOffset}天)`;
            }
        }

        // 触发地图更新
        this.onTimeChange(dayOffset);
    },

    onTimeChange(dayOffset) {
        if (!AppState.stations.length) return;

        if (dayOffset <= 0) {
            // 历史/当前 — 恢复原始数据和标记颜色
            this._restoreOriginal();
            if (typeof updateMapMarkers === 'function') {
                updateMapMarkers();
            }
            return;
        }

        // 预测模式: 根据预测值更新站点颜色
        const predIdx = dayOffset - 1;
        AppState.stations.forEach(station => {
            const pred = station.predictions?.[predIdx];
            if (!pred) return;

            // 备份原始数据（首次进入预测模式时）
            if (!station._origCurrent) {
                station._origCurrent = { ...station.current };
                station._origWqLevel = { ...station.water_quality_level };
            }

            station.current = { ...station._origCurrent, ...pred.values };

            // 重新计算水质等级
            const codmn = pred.values?.codmn || station._origCurrent.codmn || 0;
            const nh3n = pred.values?.nh3n || station._origCurrent.nh3n || 0;
            const tp = pred.values?.tp || station._origCurrent.tp || 0;
            if (codmn <= 2 && nh3n <= 0.15 && tp <= 0.02) {
                station.water_quality_level = { level: 1, name: 'I类', color: '#22c55e' };
            } else if (codmn <= 4 && nh3n <= 0.5 && tp <= 0.1) {
                station.water_quality_level = { level: 2, name: 'II类', color: '#84cc16' };
            } else if (codmn <= 6 && nh3n <= 1.0 && tp <= 0.2) {
                station.water_quality_level = { level: 3, name: 'III类', color: '#eab308' };
            } else if (codmn <= 10 && nh3n <= 1.5 && tp <= 0.3) {
                station.water_quality_level = { level: 4, name: 'IV类', color: '#f97316' };
            } else if (codmn <= 15 && nh3n <= 2.0 && tp <= 0.4) {
                station.water_quality_level = { level: 5, name: 'V类', color: '#ef4444' };
            } else {
                station.water_quality_level = { level: 6, name: '劣V类', color: '#991b1b' };
            }
        });

        if (typeof updateMapMarkers === 'function') {
            updateMapMarkers();
        }
    },

    /** 恢复所有站点的原始数据 */
    _restoreOriginal() {
        AppState.stations.forEach(station => {
            if (station._origCurrent) {
                station.current = station._origCurrent;
                delete station._origCurrent;
            }
            if (station._origWqLevel) {
                station.water_quality_level = station._origWqLevel;
                delete station._origWqLevel;
            }
        });
    },

    togglePlay() {
        if (this.playing) {
            this.stop();
        } else {
            this.play();
        }
    },

    play() {
        this.playing = true;
        if (this.playBtn) this.playBtn.innerHTML = '&#9646;&#9646;';

        if (this.currentStep >= this.totalSteps) {
            this.currentStep = 0;
            this.slider.value = 0;
        }

        this.playTimer = setInterval(() => {
            this.currentStep++;
            if (this.currentStep > this.totalSteps) {
                this.stop();
                return;
            }
            this.slider.value = this.currentStep;
            this.update();
        }, 800);
    },

    stop() {
        this.playing = false;
        if (this.playBtn) this.playBtn.innerHTML = '&#9654;';
        if (this.playTimer) {
            clearInterval(this.playTimer);
            this.playTimer = null;
        }

        // 恢复原始数据（包括 water_quality_level）
        this._restoreOriginal();
    }
};

// 初始化由 app.js 的 init() 末尾调用，不再用 setTimeout
