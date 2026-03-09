fn main() {
    println!("cargo:rerun-if-changed=installer.rc");
    println!("cargo:rerun-if-changed=icons/decomp.ico");

    #[cfg(target_os = "windows")]
    {
        embed_resource::compile("installer.rc", embed_resource::NONE)
            .manifest_optional()
            .expect("failed to compile Windows installer resources");
    }
}
