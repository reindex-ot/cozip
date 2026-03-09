#![cfg_attr(target_os = "windows", windows_subsystem = "windows")]

mod i18n;

use std::time::Duration;

use gpui::{
    App, AppContext, Application, Bounds, Context, FontWeight, IntoElement, ParentElement,
    Render, SharedString, Styled, Timer, TitlebarOptions, Window, WindowBounds, WindowOptions, div,
    prelude::*, px, rgb, size,
};

use crate::i18n::I18n;

const INSTALL_DIR: &str = r"C:\Program Files\CoZip";
const LICENSE_TEXT: &str = include_str!("../../../LICENSE");

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum InstallerStep {
    License,
    Options,
    Installing,
    Complete,
}

struct InstallerApp {
    i18n: I18n,
    step: InstallerStep,
    license_accepted: bool,
    add_explorer_menu: bool,
    install_running: bool,
    install_progress: u8,
}

impl InstallerApp {
    fn new(i18n: I18n) -> Self {
        Self {
            i18n,
            step: InstallerStep::License,
            license_accepted: false,
            add_explorer_menu: true,
            install_running: false,
            install_progress: 0,
        }
    }

    fn t(&self, key: &str) -> SharedString {
        self.i18n.text(key).to_owned().into()
    }

    fn can_go_back(&self) -> bool {
        matches!(self.step, InstallerStep::Options)
    }

    fn can_continue(&self) -> bool {
        match self.step {
            InstallerStep::License => self.license_accepted,
            InstallerStep::Options => !self.install_running,
            InstallerStep::Installing => false,
            InstallerStep::Complete => true,
        }
    }

    fn continue_label(&self) -> SharedString {
        match self.step {
            InstallerStep::Options => self.t("buttons.install"),
            InstallerStep::Complete => self.t("buttons.finish"),
            InstallerStep::License | InstallerStep::Installing => self.t("buttons.next"),
        }
    }

    fn footer_text(&self) -> SharedString {
        match self.step {
            InstallerStep::License => self.t("footer.license"),
            InstallerStep::Options => self.t("footer.options"),
            InstallerStep::Installing => self.t("footer.installing"),
            InstallerStep::Complete => self.t("footer.complete"),
        }
    }

    fn next(&mut self, window: &mut Window, cx: &mut Context<Self>) {
        match self.step {
            InstallerStep::License if self.license_accepted => {
                self.step = InstallerStep::Options;
            }
            InstallerStep::Options => self.start_install(window, cx),
            InstallerStep::Complete => cx.quit(),
            InstallerStep::Installing | InstallerStep::License => {}
        }
    }

    fn back(&mut self) {
        if self.step == InstallerStep::Options {
            self.step = InstallerStep::License;
        }
    }

    fn start_install(&mut self, window: &mut Window, cx: &mut Context<Self>) {
        if self.install_running {
            return;
        }

        self.step = InstallerStep::Installing;
        self.install_running = true;
        self.install_progress = 0;

        let entity = cx.entity().clone();
        window
            .spawn(cx, async move |cx| {
                for percent in 0..=100_u8 {
                    Timer::after(Duration::from_millis(28)).await;
                    let _ = entity.update(cx, |this, _| {
                        this.install_progress = percent;
                        if percent == 100 {
                            this.install_running = false;
                            this.step = InstallerStep::Complete;
                        }
                    });
                }
            })
            .detach();
    }

    fn install_status(&self) -> SharedString {
        let key = match self.install_progress {
            0..=14 => "install.status_preparing",
            15..=49 => "install.status_extracting",
            50..=69 => "install.status_shortcuts",
            70..=89 => "install.status_explorer",
            90..=99 => "install.status_finalizing",
            _ => "install.status_completed",
        };
        self.t(key)
    }

    fn header(&self) -> impl IntoElement {
        div()
            .px_6()
            .py_5()
            .bg(rgb(0xffffff))
            .rounded_t_xl()
            .border_b_1()
            .border_color(rgb(0xd4d4d8))
            .gap_1()
            .flex()
            .flex_col()
            .child(
                div()
                    .text_xl()
                    .font_weight(FontWeight::BOLD)
                    .text_color(rgb(0x111827))
                    .child(self.t("header.title")),
            )
            .child(
                div()
                    .text_sm()
                    .text_color(rgb(0x4b5563))
                    .child(self.t("header.subtitle")),
            )
    }

    fn content_header(&self, title: SharedString, description: SharedString) -> impl IntoElement {
        div()
            .gap_1()
            .flex()
            .flex_col()
            .child(
                div()
                    .text_lg()
                    .font_weight(FontWeight::SEMIBOLD)
                    .text_color(rgb(0x111827))
                    .child(title),
            )
            .child(
                div()
                    .text_sm()
                    .text_color(rgb(0x4b5563))
                    .child(description),
            )
    }

    fn license_screen(&self, cx: &mut Context<Self>) -> impl IntoElement {
        div()
            .w_full()
            .h_full()
            .min_h(px(0.0))
            .gap_4()
            .flex()
            .flex_col()
            .child(self.content_header(self.t("license.title"), self.t("license.description")))
            .child(
                div()
                    .id("license-scroll")
                    .w_full()
                    .flex_grow()
                    .min_h(px(120.0))
                    .bg(rgb(0xffffff))
                    .border_1()
                    .rounded_lg()
                    .border_color(rgb(0xd4d4d8))
                    .overflow_y_scroll()
                    .p_4()
                    .text_xs()
                    .text_color(rgb(0x111827))
                    .children(
                        LICENSE_TEXT
                            .lines()
                            .map(|line| div().child(line.to_string())),
                    ),
            )
            .child(self.checkbox(
                "license-accept",
                self.license_accepted,
                self.t("license.accept"),
                |this| {
                    this.license_accepted = !this.license_accepted;
                },
                cx,
            ))
    }

    fn options_screen(&self, cx: &mut Context<Self>) -> impl IntoElement {
        div()
            .w_full()
            .h_full()
            .min_h(px(0.0))
            .gap_4()
            .flex()
            .flex_col()
            .child(self.content_header(self.t("options.title"), self.t("options.description")))
            .child(value_row(self.t("common.install_dir"), INSTALL_DIR.into()))
            .child(value_row(
                self.t("common.start_menu"),
                self.t("common.start_menu_value"),
            ))
            .child(self.checkbox(
                "explorer-menu",
                self.add_explorer_menu,
                self.t("options.explorer_menu"),
                |this| {
                    this.add_explorer_menu = !this.add_explorer_menu;
                },
                cx,
            ))
    }

    fn install_screen(&self) -> impl IntoElement {
        div()
            .w_full()
            .h_full()
            .min_h(px(0.0))
            .gap_4()
            .flex()
            .flex_col()
            .child(self.content_header(self.t("install.title"), self.t("install.description")))
            .child(
                div()
                    .w_full()
                    .bg(rgb(0xffffff))
                    .border_1()
                    .rounded_lg()
                    .border_color(rgb(0xd4d4d8))
                    .p_5()
                    .gap_4()
                    .flex()
                    .flex_col()
                    .child(simple_progress_bar(self.install_progress))
                    .child(
                        div()
                            .flex()
                            .justify_between()
                            .items_center()
                            .child(
                                div()
                                    .text_sm()
                                    .text_color(rgb(0x111827))
                                    .child(self.install_status()),
                            )
                            .child(
                                div()
                                    .text_sm()
                                    .font_weight(FontWeight::BOLD)
                                    .text_color(rgb(0x111827))
                                    .child(format!("{}%", self.install_progress)),
                            ),
                    ),
            )
    }

    fn complete_screen(&self) -> impl IntoElement {
        div()
            .w_full()
            .h_full()
            .min_h(px(0.0))
            .gap_4()
            .flex()
            .flex_col()
            .child(self.content_header(self.t("complete.title"), self.t("complete.description")))
            .child(value_row(self.t("common.install_dir"), INSTALL_DIR.into()))
            .child(value_row(
                self.t("options.explorer_menu"),
                if self.add_explorer_menu {
                    self.t("complete.menu_enabled")
                } else {
                    self.t("complete.menu_disabled")
                },
            ))
            .child(value_row("Status".into(), self.t("complete.state")))
    }

    fn checkbox(
        &self,
        id: &'static str,
        checked: bool,
        label: SharedString,
        on_click: impl Fn(&mut Self) + 'static,
        cx: &mut Context<Self>,
    ) -> impl IntoElement {
        div()
            .id(SharedString::from(id))
            .w_full()
            .px_4()
            .py_3()
            .bg(rgb(0xffffff))
            .border_1()
            .rounded_lg()
            .border_color(rgb(0xd4d4d8))
            .flex()
            .items_center()
            .gap_3()
            .cursor_pointer()
            .hover(|style| style.bg(rgb(0xf9fafb)))
            .on_click(cx.listener(move |this, _, _, _| {
                on_click(this);
            }))
            .child(
                div()
                    .w(px(18.0))
                    .h(px(18.0))
                    .border_1()
                    .rounded_sm()
                    .border_color(if checked { rgb(0x2563eb) } else { rgb(0x6b7280) })
                    .bg(if checked {
                        rgb(0x2563eb)
                    } else {
                        rgb(0xffffff)
                    })
                    .flex()
                    .items_center()
                    .justify_center()
                    .text_xs()
                    .font_weight(FontWeight::BOLD)
                    .text_color(rgb(0xffffff))
                    .child(if checked { "✓" } else { "" }),
            )
            .child(
                div()
                    .text_sm()
                    .text_color(rgb(0x111827))
                    .child(label),
            )
    }

    fn button(
        &self,
        id: &'static str,
        label: SharedString,
        enabled: bool,
        primary: bool,
        on_click: impl Fn(&mut Self, &mut Window, &mut Context<Self>) + 'static,
        cx: &mut Context<Self>,
    ) -> impl IntoElement {
        let mut button = div()
            .id(SharedString::from(id))
            .min_w(px(92.0))
            .px_4()
            .py_2()
            .border_1()
            .rounded_lg()
            .border_color(if enabled {
                rgb(0xa1a1aa)
            } else {
                rgb(0xd4d4d8)
            })
            .bg(if primary && enabled {
                rgb(0xf3f4f6)
            } else {
                rgb(0xffffff)
            })
            .text_sm()
            .text_color(if enabled {
                rgb(0x111827)
            } else {
                rgb(0x9ca3af)
            })
            .font_weight(FontWeight::MEDIUM)
            .child(label);

        if enabled {
            button = button
                .cursor_pointer()
                .hover(|style| style.bg(rgb(0xf3f4f6)))
                .on_click(cx.listener(move |this, _, window, cx| {
                    on_click(this, window, cx);
                }));
        }

        button
    }
}

impl Render for InstallerApp {
    fn render(&mut self, _window: &mut Window, cx: &mut Context<Self>) -> impl IntoElement {
        let content = match self.step {
            InstallerStep::License => self.license_screen(cx).into_any_element(),
            InstallerStep::Options => self.options_screen(cx).into_any_element(),
            InstallerStep::Installing => self.install_screen().into_any_element(),
            InstallerStep::Complete => self.complete_screen().into_any_element(),
        };

        div()
            .size_full()
            .bg(rgb(0xe5e7eb))
            .p_6()
            .child(
                div()
                    .size_full()
                    .bg(rgb(0xffffff))
                    .border_1()
                    .rounded_xl()
                    .border_color(rgb(0xa1a1aa))
                    .flex()
                    .flex_col()
                    .child(self.header())
                    .child(
                        div()
                            .flex_grow()
                            .min_h(px(0.0))
                            .px_6()
                            .pt_5()
                            .pb_8()
                            .bg(rgb(0xffffff))
                            .child(div().w_full().h_full().min_h(px(0.0)).pb_2().child(content)),
                    )
                    .child(
                        div()
                            .px_6()
                            .py_5()
                            .rounded_b_xl()
                            .border_t_1()
                            .border_color(rgb(0xd4d4d8))
                            .bg(rgb(0xf9fafb))
                            .flex()
                            .justify_between()
                            .items_center()
                            .child(
                                div()
                                    .max_w(px(360.0))
                                    .pr_4()
                                    .text_sm()
                                    .text_color(rgb(0x6b7280))
                                    .child(self.footer_text()),
                            )
                            .child(
                                div()
                                    .flex()
                                    .gap_2()
                                    .child(self.button(
                                        "back-button",
                                        self.t("buttons.back"),
                                        self.can_go_back(),
                                        false,
                                        |this, _, _| this.back(),
                                        cx,
                                    ))
                                    .child(self.button(
                                        "next-button",
                                        self.continue_label(),
                                        self.can_continue(),
                                        true,
                                        |this, window, cx| this.next(window, cx),
                                        cx,
                                    )),
                            ),
                    ),
            )
    }
}

fn value_row(label: SharedString, value: SharedString) -> impl IntoElement {
    div()
        .w_full()
        .bg(rgb(0xffffff))
        .border_1()
        .rounded_lg()
        .border_color(rgb(0xd4d4d8))
        .px_4()
        .py_3()
        .flex()
        .justify_between()
        .gap_4()
        .child(
            div()
                .text_sm()
                .text_color(rgb(0x4b5563))
                .child(label),
        )
        .child(
            div()
                .text_sm()
                .text_color(rgb(0x111827))
                .child(value),
        )
}

fn simple_progress_bar(progress: u8) -> impl IntoElement {
    let width = (f32::from(progress) / 100.0) * 560.0;
    div()
        .w_full()
        .max_w(px(560.0))
        .h(px(18.0))
        .border_1()
        .rounded_full()
        .overflow_hidden()
        .border_color(rgb(0xa1a1aa))
        .bg(rgb(0xffffff))
        .child(
            div()
                .h_full()
                .w(px(width))
                .bg(rgb(0x2563eb))
                .rounded_full(),
        )
}

fn main() {
    let i18n = I18n::load();
    let title = SharedString::from(i18n.text("window.title").to_string());

    Application::new().run(move |cx: &mut App| {
        let bounds = Bounds::centered(None, size(px(720.0), px(560.0)), cx);
        let i18n = i18n.clone();
        let title = title.clone();
        cx.open_window(
            WindowOptions {
                window_bounds: Some(WindowBounds::Windowed(bounds)),
                titlebar: Some(TitlebarOptions {
                    title: Some(title),
                    ..Default::default()
                }),
                app_id: Some("cozip-installer".to_string()),
                window_min_size: Some(size(px(680.0), px(520.0))),
                ..Default::default()
            },
            move |_, cx| cx.new(|_| InstallerApp::new(i18n.clone())),
        )
        .expect("failed to open installer window");
        cx.activate(true);
    });
}
