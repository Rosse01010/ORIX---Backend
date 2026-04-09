import { ToastContainer } from "react-toastify";
import RouterProvider from "./router";

export default function App() {
  return (
    <>
      <RouterProvider />
      <ToastContainer
        position="top-right"
        theme="dark"
        autoClose={4000}
        pauseOnHover
        newestOnTop
      />
    </>
  );
}
