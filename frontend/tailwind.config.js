/** @type {import('tailwindcss').Config} */
export default {
  content: ["./index.html", "./src/**/*.{js,ts,jsx,tsx}"],
  theme: {
    extend: {
      screens: {
        xs: "475px",
      },
      colors: {
        orix: {
          bg: "#09090b",
          surface: "#18181b",
          border: "#27272a",
          accent: "#3b82f6",
          danger: "#dc2626",
          success: "#16a34a",
        },
      },
      keyframes: {
        "slide-in": {
          from: { transform: "translateX(100%)", opacity: "0" },
          to:   { transform: "translateX(0)",    opacity: "1" },
        },
      },
      animation: {
        "slide-in": "slide-in 0.25s ease-out forwards",
      },
    },
  },
  plugins: [],
};
