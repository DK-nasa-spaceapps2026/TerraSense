import type { Config } from "tailwindcss";

export default {
  content: ["./app/**/*.{ts,tsx}", "./components/**/*.{ts,tsx}"],
  theme: {
    extend: {
      colors: {
        // Claude warm-charcoal dark palette + TerraSense green accent
        soil: {
          900: "#141413",
          800: "#1E1E1C",
          700: "#2A2A27",
          600: "#383834",
        },
        claude: {
          cream: "#FAF9F5",
          muted: "#B0AEA5",
          terracotta: "#D97757",
        },
      },
    },
  },
  plugins: [],
} satisfies Config;
