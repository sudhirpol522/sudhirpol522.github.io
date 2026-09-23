/** Static export so the site can be hosted on GitHub Pages. */
const nextConfig = {
  output: "export",
  trailingSlash: true,
  images: { unoptimized: true },
  devIndicators: false,
};

export default nextConfig;
