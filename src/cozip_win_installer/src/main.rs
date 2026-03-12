#![cfg_attr(target_os = "windows", windows_subsystem = "windows")]

mod i18n;

#[cfg(target_os = "windows")]
use std::ffi::OsStr;
use std::fs::{self, File};
use std::io::{Read, Write};
#[cfg(target_os = "windows")]
use std::os::windows::ffi::OsStrExt;
use std::path::{Path, PathBuf};
use std::process::Command;
use std::sync::mpsc;
use std::thread;
use std::time::Duration;

use gpui::{
    App, AppContext, Application, Bounds, Context, FontWeight, IntoElement, ParentElement,
    Render, SharedString, Styled, Timer, TitlebarOptions, Window, WindowBounds, WindowOptions, div,
    prelude::*, px, rgb, size,
};
use reqwest::blocking::Client;
use serde::Deserialize;
use zip::ZipArchive;

use crate::i18n::I18n;

#[cfg(target_os = "windows")]
use windows_sys::Win32::Foundation::{CloseHandle, HANDLE, HWND};
#[cfg(target_os = "windows")]
use windows_sys::Win32::Security::{GetTokenInformation, TOKEN_ELEVATION, TOKEN_QUERY, TokenElevation};
#[cfg(target_os = "windows")]
use windows_sys::Win32::System::Threading::{GetCurrentProcess, OpenProcessToken};
#[cfg(target_os = "windows")]
use windows_sys::Win32::UI::Shell::ShellExecuteW;

const INSTALL_DIR: &str = r"C:\Program Files\CoZip";
const COZIP_DESKTOP_EXE_PATH: &str = r"C:\Program Files\CoZip\cozip_desktop.exe";
const COMP_ICON_PATH: &str = r"C:\Program Files\CoZip\icons\comp.ico";
const DECOMP_ICON_PATH: &str = r"C:\Program Files\CoZip\icons\decomp.ico";
const COZIP_FILE_PROG_ID: &str = "CoZip.Archive";
const RELEASES_API_URL: &str = "https://api.github.com/repos/bea4dev/cozip/releases";
const RELEASE_ASSET_NAME: &str = "pack.zip";
const DOWNLOAD_PROGRESS_END: u8 = 59;
const EXTRACT_PROGRESS_END: u8 = 89;
const REGISTRY_PROGRESS_END: u8 = 100;
const LICENSE_TEXT: &str = include_str!("../../../LICENSE");

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum InstallerStep {
    License,
    Options,
    Installing,
    Complete,
}

enum InstallWorkerMessage {
    Progress(u8),
    Finished(Result<(), String>),
}

#[derive(Debug, Deserialize)]
struct GithubRelease {
    draft: bool,
    prerelease: bool,
    assets: Vec<GithubReleaseAsset>,
}

#[derive(Clone, Debug, Deserialize)]
struct GithubReleaseAsset {
    name: String,
    browser_download_url: String,
    size: u64,
}

struct InstallerApp {
    i18n: I18n,
    step: InstallerStep,
    license_accepted: bool,
    add_explorer_menu: bool,
    install_running: bool,
    install_progress: u8,
    install_note: Option<String>,
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
            install_note: None,
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
        self.install_note = None;

        let add_explorer_menu = self.add_explorer_menu;
        let (sender, receiver) = mpsc::channel::<InstallWorkerMessage>();
        thread::spawn(move || {
            let result = perform_install(add_explorer_menu, |progress| {
                let _ = sender.send(InstallWorkerMessage::Progress(progress));
            });
            let _ = sender.send(InstallWorkerMessage::Finished(result));
        });

        let entity = cx.entity().clone();
        window
            .spawn(cx, async move |cx| {
                loop {
                    let mut latest_progress = None;
                    let mut finished = None;
                    while let Ok(message) = receiver.try_recv() {
                        match message {
                            InstallWorkerMessage::Progress(progress) => {
                                latest_progress = Some(progress);
                            }
                            InstallWorkerMessage::Finished(result) => {
                                finished = Some(result);
                                break;
                            }
                        }
                    }

                    if let Some(progress) = latest_progress {
                        let _ = entity.update(cx, |this, _| {
                            this.install_progress = progress;
                        });
                    }

                    if let Some(result) = finished {
                        let _ = entity.update(cx, |this, _| {
                            this.install_progress = 100;
                            this.install_running = false;
                            this.step = InstallerStep::Complete;
                            this.install_note = result.err();
                        });
                        break;
                    }

                    Timer::after(Duration::from_millis(120)).await;
                }
            })
            .detach();
    }

    fn install_status(&self) -> SharedString {
        let key = match self.install_progress {
            0..=9 => "install.status_preparing",
            10..=59 => "install.status_downloading",
            60..=89 => "install.status_extracting",
            90..=99 => "install.status_registering",
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
            .when_some(self.install_note.as_ref(), |this, note| {
                this.child(value_row(self.t("complete.note"), note.clone().into()))
            })
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

fn perform_install(
    add_explorer_menu: bool,
    mut on_progress: impl FnMut(u8),
) -> Result<(), String> {
    let mut last_progress = 0_u8;
    let mut report_progress = |progress: u8| {
        if progress > last_progress {
            last_progress = progress;
            on_progress(progress);
        }
    };

    report_progress(1);

    let client = Client::builder()
        .timeout(Duration::from_secs(120))
        .build()
        .map_err(|error| format!("failed to initialize http client: {error}"))?;
    let asset = fetch_latest_release_asset(&client)?;

    let temp_zip_path = installer_temp_zip_path();
    let download_result = download_release_asset(&client, &asset, &temp_zip_path, |written, total| {
        report_progress(scale_progress(5, DOWNLOAD_PROGRESS_END, written, total));
    });
    if let Err(error) = download_result {
        let _ = fs::remove_file(&temp_zip_path);
        return Err(error);
    }

    let extract_result = extract_pack_archive(&temp_zip_path, Path::new(INSTALL_DIR), |written, total| {
        report_progress(scale_progress(
            DOWNLOAD_PROGRESS_END + 1,
            EXTRACT_PROGRESS_END,
            written,
            total,
        ));
    });
    let _ = fs::remove_file(&temp_zip_path);
    extract_result?;

    install_archive_file_associations()?;
    report_progress(94);
    if add_explorer_menu {
        install_explorer_menu_entries()?;
    }
    report_progress(REGISTRY_PROGRESS_END);
    Ok(())
}

fn fetch_latest_release_asset(client: &Client) -> Result<GithubReleaseAsset, String> {
    let releases = client
        .get(RELEASES_API_URL)
        .header(reqwest::header::USER_AGENT, "cozip-win-installer")
        .header(reqwest::header::ACCEPT, "application/vnd.github+json")
        .send()
        .map_err(|error| format!("failed to query GitHub releases: {error}"))?
        .error_for_status()
        .map_err(|error| format!("GitHub releases request failed: {error}"))?
        .json::<Vec<GithubRelease>>()
        .map_err(|error| format!("failed to parse GitHub releases: {error}"))?;

    releases
        .into_iter()
        .filter(|release| !release.prerelease && !release.draft)
        .find_map(|release| {
            release
                .assets
                .into_iter()
                .find(|asset| asset.name.eq_ignore_ascii_case(RELEASE_ASSET_NAME))
        })
        .ok_or_else(|| "latest stable release does not contain pack.zip".to_string())
}

fn download_release_asset(
    client: &Client,
    asset: &GithubReleaseAsset,
    destination: &Path,
    mut on_progress: impl FnMut(u64, u64),
) -> Result<(), String> {
    if let Some(parent) = destination.parent() {
        fs::create_dir_all(parent).map_err(|error| {
            format!("failed to create temporary directory {}: {error}", parent.display())
        })?;
    }

    let mut response = client
        .get(&asset.browser_download_url)
        .header(reqwest::header::USER_AGENT, "cozip-win-installer")
        .send()
        .map_err(|error| format!("failed to download release asset: {error}"))?
        .error_for_status()
        .map_err(|error| format!("release asset download failed: {error}"))?;

    let mut file = File::create(destination)
        .map_err(|error| format!("failed to create {}: {error}", destination.display()))?;
    let total = response.content_length().unwrap_or(asset.size).max(1);
    let mut written = 0_u64;
    let mut buffer = vec![0_u8; 1024 * 1024];
    on_progress(0, total);

    loop {
        let read = response
            .read(&mut buffer)
            .map_err(|error| format!("failed while downloading pack.zip: {error}"))?;
        if read == 0 {
            break;
        }
        file.write_all(&buffer[..read])
            .map_err(|error| format!("failed to write {}: {error}", destination.display()))?;
        written = written.saturating_add(read as u64);
        on_progress(written.min(total), total);
    }

    file.flush()
        .map_err(|error| format!("failed to flush {}: {error}", destination.display()))?;
    on_progress(total, total);
    Ok(())
}

fn extract_pack_archive(
    archive_path: &Path,
    install_dir: &Path,
    mut on_progress: impl FnMut(u64, u64),
) -> Result<(), String> {
    if install_dir.exists() {
        fs::remove_dir_all(install_dir)
            .map_err(|error| format!("failed to clear {}: {error}", install_dir.display()))?;
    }
    fs::create_dir_all(install_dir)
        .map_err(|error| format!("failed to create {}: {error}", install_dir.display()))?;

    let file = File::open(archive_path)
        .map_err(|error| format!("failed to open {}: {error}", archive_path.display()))?;
    let mut archive = ZipArchive::new(file)
        .map_err(|error| format!("failed to read {}: {error}", archive_path.display()))?;

    let mut total_bytes = 0_u64;
    for index in 0..archive.len() {
        let entry = archive
            .by_index(index)
            .map_err(|error| format!("failed to inspect archive entry #{index}: {error}"))?;
        total_bytes = total_bytes.saturating_add(entry.size().max(1));
    }
    total_bytes = total_bytes.max(1);
    on_progress(0, total_bytes);

    let mut processed_bytes = 0_u64;
    let mut buffer = vec![0_u8; 1024 * 1024];
    for index in 0..archive.len() {
        let mut entry = archive
            .by_index(index)
            .map_err(|error| format!("failed to read archive entry #{index}: {error}"))?;
        let Some(relative_path) = entry.enclosed_name().map(PathBuf::from) else {
            return Err(format!("archive entry #{index} has an invalid path"));
        };
        let output_path = install_dir.join(relative_path);

        if entry.is_dir() {
            fs::create_dir_all(&output_path)
                .map_err(|error| format!("failed to create {}: {error}", output_path.display()))?;
            processed_bytes = processed_bytes.saturating_add(1);
            on_progress(processed_bytes.min(total_bytes), total_bytes);
            continue;
        }

        if let Some(parent) = output_path.parent() {
            fs::create_dir_all(parent)
                .map_err(|error| format!("failed to create {}: {error}", parent.display()))?;
        }

        let mut output = File::create(&output_path)
            .map_err(|error| format!("failed to create {}: {error}", output_path.display()))?;
        loop {
            let read = entry
                .read(&mut buffer)
                .map_err(|error| format!("failed to extract {}: {error}", output_path.display()))?;
            if read == 0 {
                break;
            }
            output
                .write_all(&buffer[..read])
                .map_err(|error| format!("failed to write {}: {error}", output_path.display()))?;
            processed_bytes = processed_bytes.saturating_add(read as u64);
            on_progress(processed_bytes.min(total_bytes), total_bytes);
        }
    }

    on_progress(total_bytes, total_bytes);
    Ok(())
}

fn scale_progress(start: u8, end: u8, completed: u64, total: u64) -> u8 {
    if end <= start || total == 0 {
        return end;
    }
    let span = u64::from(end - start);
    let bounded = completed.min(total);
    let offset = (bounded.saturating_mul(span) / total) as u8;
    start.saturating_add(offset).min(end)
}

fn installer_temp_zip_path() -> PathBuf {
    std::env::temp_dir().join(format!("cozip-installer-pack-{}.zip", std::process::id()))
}

#[cfg(target_os = "windows")]
fn install_explorer_menu_entries() -> Result<(), String> {
    let compress_root = r"HKCU\Software\Classes\AllFilesystemObjects\shell\CozipCompress";
    let extract_root = r"HKCU\Software\Classes\AllFilesystemObjects\shell\CozipExtract";
    let compress_subcommands = r"HKCU\Software\Classes\CoZip.ContextMenus\Compress";
    let extract_subcommands = r"HKCU\Software\Classes\CoZip.ContextMenus\Extract";

    let _ = reg_delete_tree(compress_subcommands);
    let _ = reg_delete_tree(extract_subcommands);

    reg_add_value(compress_root, Some("MUIVerb"), "圧縮")?;
    reg_add_value(compress_root, Some("Icon"), COMP_ICON_PATH)?;
    reg_add_value(compress_root, Some("MultiSelectModel"), "Player")?;
    reg_add_value(
        compress_root,
        Some("ExtendedSubCommandsKey"),
        r"CoZip.ContextMenus\Compress",
    )?;

    reg_add_value(
        &format!(r"{compress_subcommands}\shell\01_zip_gpu"),
        Some("MUIVerb"),
        "zip (CPU + GPU)",
    )?;
    reg_add_value(
        &format!(r"{compress_subcommands}\shell\01_zip_gpu\command"),
        None,
        &format!(r#""{COZIP_DESKTOP_EXE_PATH}" compress --format zip --hybrid %*"#),
    )?;

    reg_add_value(
        &format!(r"{compress_subcommands}\shell\02_cozip_gpu"),
        Some("MUIVerb"),
        "cozip (試験的) (CPU + GPU)",
    )?;
    reg_add_value(
        &format!(r"{compress_subcommands}\shell\02_cozip_gpu\command"),
        None,
        &format!(r#""{COZIP_DESKTOP_EXE_PATH}" compress --format cozip --hybrid %*"#),
    )?;

    reg_add_value(
        &format!(r"{compress_subcommands}\shell\03_details"),
        Some("MUIVerb"),
        "詳細設定",
    )?;
    reg_add_value(
        &format!(r"{compress_subcommands}\shell\03_details\command"),
        None,
        &format!(r#""{COZIP_DESKTOP_EXE_PATH}" ui compress-details %*"#),
    )?;

    reg_add_value(extract_root, Some("MUIVerb"), "解凍")?;
    reg_add_value(extract_root, Some("Icon"), DECOMP_ICON_PATH)?;
    reg_add_value(extract_root, Some("MultiSelectModel"), "Player")?;
    reg_add_value(
        extract_root,
        Some("ExtendedSubCommandsKey"),
        r"CoZip.ContextMenus\Extract",
    )?;

    reg_add_value(
        &format!(r"{extract_subcommands}\shell\01_extract_here"),
        Some("MUIVerb"),
        "ここに解凍",
    )?;
    reg_add_value(
        &format!(r"{extract_subcommands}\shell\01_extract_here\command"),
        None,
        &format!(r#""{COZIP_DESKTOP_EXE_PATH}" extract --here "%1" %*"#),
    )?;

    reg_add_value(
        &format!(r"{extract_subcommands}\shell\02_details"),
        Some("MUIVerb"),
        "詳細設定",
    )?;
    reg_add_value(
        &format!(r"{extract_subcommands}\shell\02_details\command"),
        None,
        &format!(r#""{COZIP_DESKTOP_EXE_PATH}" ui extract-details "%1" %*"#),
    )?;

    Ok(())
}

#[cfg(target_os = "windows")]
fn install_archive_file_associations() -> Result<(), String> {
    let extension_key = r"HKCU\Software\Classes\.cozip";
    let prog_id_key = r"HKCU\Software\Classes\CoZip.Archive";

    reg_add_value(extension_key, None, COZIP_FILE_PROG_ID)?;
    reg_add_value(prog_id_key, None, "CoZip Archive")?;
    reg_add_value(
        &format!(r"{prog_id_key}\DefaultIcon"),
        None,
        DECOMP_ICON_PATH,
    )?;
    reg_add_value(
        &format!(r"{prog_id_key}\shell\open\command"),
        None,
        &format!(r#""{COZIP_DESKTOP_EXE_PATH}" extract --here "%1""#),
    )?;

    Ok(())
}

#[cfg(not(target_os = "windows"))]
fn install_archive_file_associations() -> Result<(), String> {
    Ok(())
}

#[cfg(not(target_os = "windows"))]
fn install_explorer_menu_entries() -> Result<(), String> {
    Ok(())
}

#[cfg(target_os = "windows")]
fn reg_add_value(key: &str, value_name: Option<&str>, data: &str) -> Result<(), String> {
    let mut command = Command::new("reg");
    command.args(["add", key, "/t", "REG_SZ", "/d", data, "/f"]);
    match value_name {
        Some(name) => {
            command.args(["/v", name]);
        }
        None => {
            command.arg("/ve");
        }
    }

    let output = command
        .output()
        .map_err(|error| format!("reg add failed for {key}: {error}"))?;

    if output.status.success() {
        Ok(())
    } else {
        let stderr = String::from_utf8_lossy(&output.stderr).trim().to_string();
        let stdout = String::from_utf8_lossy(&output.stdout).trim().to_string();
        let detail = if !stderr.is_empty() { stderr } else { stdout };
        Err(format!("registry write failed for {key}: {detail}"))
    }
}

#[cfg(target_os = "windows")]
fn reg_delete_tree(key: &str) -> Result<(), String> {
    let output = Command::new("reg")
        .args(["delete", key, "/f"])
        .output()
        .map_err(|error| format!("reg delete failed for {key}: {error}"))?;

    if output.status.success() {
        Ok(())
    } else {
        let stderr = String::from_utf8_lossy(&output.stderr).trim().to_string();
        let stdout = String::from_utf8_lossy(&output.stdout).trim().to_string();
        let detail = if !stderr.is_empty() { stderr } else { stdout };
        if detail.contains("unable to find") || detail.contains("指定されたレジストリ") {
            Ok(())
        } else {
            Err(format!("registry delete failed for {key}: {detail}"))
        }
    }
}

#[cfg(target_os = "windows")]
fn ensure_elevated() -> Result<(), String> {
    if is_process_elevated()? {
        return Ok(());
    }

    let exe_path = std::env::current_exe().map_err(|error| format!("current_exe failed: {error}"))?;
    let current_dir = std::env::current_dir().map_err(|error| format!("current_dir failed: {error}"))?;
    let args = std::env::args_os()
        .skip(1)
        .map(|arg| quote_windows_arg(&arg))
        .collect::<Vec<_>>()
        .join(" ");

    let exe_wide = to_wide(exe_path.as_os_str());
    let verb_wide = to_wide(OsStr::new("runas"));
    let dir_wide = to_wide(current_dir.as_os_str());
    let args_wide = if args.is_empty() {
        Vec::new()
    } else {
        to_wide(OsStr::new(&args))
    };

    let result = unsafe {
        ShellExecuteW(
            0 as HWND,
            verb_wide.as_ptr(),
            exe_wide.as_ptr(),
            if args_wide.is_empty() {
                std::ptr::null()
            } else {
                args_wide.as_ptr()
            },
            dir_wide.as_ptr(),
            1,
        )
    } as isize;

    if result <= 32 {
        return Err(format!("ShellExecuteW failed with code {result}"));
    }

    std::process::exit(0);
}

#[cfg(target_os = "windows")]
fn is_process_elevated() -> Result<bool, String> {
    let mut token: HANDLE = std::ptr::null_mut();
    let opened = unsafe { OpenProcessToken(GetCurrentProcess(), TOKEN_QUERY, &mut token) };
    if opened == 0 {
        return Err("OpenProcessToken failed".to_string());
    }

    let mut elevation = TOKEN_ELEVATION { TokenIsElevated: 0 };
    let mut returned_len = 0_u32;
    let ok = unsafe {
        GetTokenInformation(
            token,
            TokenElevation,
            &mut elevation as *mut _ as *mut _,
            std::mem::size_of::<TOKEN_ELEVATION>() as u32,
            &mut returned_len,
        )
    };
    unsafe {
        CloseHandle(token);
    }

    if ok == 0 {
        return Err("GetTokenInformation(TokenElevation) failed".to_string());
    }

    Ok(elevation.TokenIsElevated != 0)
}

#[cfg(target_os = "windows")]
fn to_wide(value: &OsStr) -> Vec<u16> {
    value.encode_wide().chain(std::iter::once(0)).collect()
}

#[cfg(target_os = "windows")]
fn quote_windows_arg(arg: &OsStr) -> String {
    let value = arg.to_string_lossy();
    if !value.contains([' ', '\t', '"']) {
        return value.into_owned();
    }

    let mut quoted = String::from("\"");
    let mut backslashes = 0_usize;
    for ch in value.chars() {
        match ch {
            '\\' => backslashes += 1,
            '"' => {
                quoted.push_str(&"\\".repeat((backslashes * 2) + 1));
                quoted.push('"');
                backslashes = 0;
            }
            _ => {
                if backslashes > 0 {
                    quoted.push_str(&"\\".repeat(backslashes));
                    backslashes = 0;
                }
                quoted.push(ch);
            }
        }
    }
    if backslashes > 0 {
        quoted.push_str(&"\\".repeat(backslashes * 2));
    }
    quoted.push('"');
    quoted
}

fn main() {
    #[cfg(target_os = "windows")]
    if let Err(error) = ensure_elevated() {
        eprintln!("failed to elevate installer: {error}");
        std::process::exit(1);
    }

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
