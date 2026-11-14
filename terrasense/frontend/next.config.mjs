const API = process.env.TERRASENSE_API ?? "http://localhost:5001";

/** @type {import('next').NextConfig} */
export default {
  reactStrictMode: true,
  // Proxy /api/* to Flask so the browser never deals with CORS.
  async rewrites() {
    return [{ source: "/api/:path*", destination: `${API}/api/:path*` }];
  },
};
