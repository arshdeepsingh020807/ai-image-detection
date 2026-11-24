import { BrowserRouter, Route, Routes } from "react-router-dom";
import { Layout } from "./components/Layout";
import { DetectPage } from "./pages/DetectPage";
import { MethodPage } from "./pages/MethodPage";
import { NotFoundPage } from "./pages/NotFoundPage";
import { PerformancePage } from "./pages/PerformancePage";

export function AppRoutes() {
  return (
    <Routes>
      <Route element={<Layout />}>
        <Route index element={<DetectPage />} />
        <Route path="performance" element={<PerformancePage />} />
        <Route path="method" element={<MethodPage />} />
        <Route path="*" element={<NotFoundPage />} />
      </Route>
    </Routes>
  );
}

export function App() {
  return (
    <BrowserRouter>
      <AppRoutes />
    </BrowserRouter>
  );
}
