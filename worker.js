// worker.js
export default {
    async fetch(req, _) {
        const url = new URL(req.url);
        // 把目标地址放在 ?url= 参数里
        const target = decodeURIComponent(url.searchParams.get('url') || '');
        if (!target) return new Response('Missing ?url', { status: 400 });
        return fetch(target, req);
    }
}