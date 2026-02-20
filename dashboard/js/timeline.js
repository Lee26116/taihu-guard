/**
 * TaihuGuard — 时间轴模块
 * 历史/预测时间导航
 */

// 当前可直接由时间序列图中的 tooltip 和标记线替代
// 此模块保留为后续扩展时间滑动条功能

const Timeline = {
    currentTime: new Date(),
    minTime: null,
    maxTime: null,

    init() {
        const now = new Date();
        this.currentTime = now;
        this.minTime = new Date(now.getTime() - 7 * 24 * 3600 * 1000);
        this.maxTime = new Date(now.getTime() + 7 * 24 * 3600 * 1000);
    },

    setTime(date) {
        this.currentTime = date;
        this.onTimeChange(date);
    },

    onTimeChange(date) {
        // 触发时间变更事件
        // 后续可用于地图动画: 显示指定时刻的全湖水质分布
        console.log('Timeline: ', date.toISOString());
    },

    formatTimeLabel(date) {
        const month = String(date.getMonth() + 1).padStart(2, '0');
        const day = String(date.getDate()).padStart(2, '0');
        const hour = String(date.getHours()).padStart(2, '0');
        return `${month}-${day} ${hour}:00`;
    }
};

// 初始化
Timeline.init();
