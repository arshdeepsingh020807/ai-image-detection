import { fireEvent, render } from "@testing-library/react";
import { MemoryRouter } from "react-router-dom";
import type { ReactElement } from "react";
import { AppRoutes } from "../App";

export const renderAt = (path: string) => render(<MemoryRouter initialEntries={[path]}><AppRoutes /></MemoryRouter>);
export const withRouter = (ui: ReactElement) => render(<MemoryRouter>{ui}</MemoryRouter>);

export function loadImage(img: HTMLElement, width: number, height: number) {
  Object.defineProperty(img, "naturalWidth", { value: width, configurable: true });
  Object.defineProperty(img, "naturalHeight", { value: height, configurable: true });
  fireEvent.load(img);
}
