import type { Config } from "tailwindcss";

const config: Config = {
  content: [
    "./app/**/*.{js,ts,jsx,tsx,mdx}",
    "./components/**/*.{js,ts,jsx,tsx,mdx}",
    "./lib/**/*.{js,ts}",
  ],
  darkMode: "class",
  theme: {
    extend: {
      fontFamily: {
        prompt: ["var(--font-prompt)"],
      },
      colors: {
        sidebar: {
          DEFAULT: "#f8f9fa",
          dark: "#111827",
        },
      },
    },
  },
  plugins: [require("@tailwindcss/typography")],
};

export default config;
