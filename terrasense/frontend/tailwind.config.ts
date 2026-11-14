import type { Config } from "tailwindcss";

export default {
  content: ["./app/**/*.{ts,tsx}", "./components/**/*.{ts,tsx}"],
  theme: {
    extend: {
      colors: { soil: { 900: "#1b1712", 800: "#262019", 700: "#352c22" } },
    },
  },
  plugins: [],
} satisfies Config;
