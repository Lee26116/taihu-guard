/**
 * TaihuGuard — 视觉效果工具
 * 数字缓动动画
 */

/**
 * 数字滚动动画
 * @param {HTMLElement} el - 目标 DOM 元素
 * @param {number} start - 起始值
 * @param {number} end - 结束值
 * @param {number} duration - 动画时长(ms)
 * @param {number} decimals - 小数位数
 */
function animateValue(el, start, end, duration, decimals) {
    if (!el) return;
    if (typeof decimals !== 'number') decimals = 0;

    const startTime = performance.now();

    function easeOutCubic(t) {
        return 1 - Math.pow(1 - t, 3);
    }

    function tick(now) {
        const elapsed = now - startTime;
        const progress = Math.min(elapsed / duration, 1);
        const eased = easeOutCubic(progress);
        const current = start + (end - start) * eased;

        el.textContent = current.toFixed(decimals);

        if (progress < 1) {
            requestAnimationFrame(tick);
        }
    }

    requestAnimationFrame(tick);
}
