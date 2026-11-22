/* psx_ios_boot.cpp — iOS start-up shim (see psx_ios_boot.h).
 *
 * Layout on the device:
 *   <App>.app/                 read-only: executable, bios/, game.toml, Frameworks/
 *   <container>/Documents/     the user's folder in the Files app (writable):
 *       *.cue + *.bin          the user's own disc image
 *       saves/                 memory cards, save states, options
 *       args.txt               optional: one extra argument per line
 *       psxrecomp.log          this run's log (previous run: psxrecomp.prev.log)
 */
#include "psx_ios_boot.h"
#ifdef PSX_IOS
#include "psx_sdl.h"
#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <string>
#include <vector>
#include <atomic>
#include <fcntl.h>
#include <unistd.h>

namespace fs = std::filesystem;

static std::vector<std::string> g_args;
static std::vector<char *> g_argv;

static bool has_flag(const std::vector<std::string> &a, const char *flag) {
    return std::find(a.begin(), a.end(), flag) != a.end();
}

extern "C" void psx_ios_prepare(int *argc, char ***argv) {
    std::error_code ec;
    const char *home = std::getenv("HOME");
    const fs::path docs = fs::path(home ? home : ".") / "Documents";
    fs::create_directories(docs / "saves", ec);
    if (chdir(docs.c_str()) != 0) std::perror("psxrecomp-ios: chdir(Documents)");

    const fs::path log = docs / "psxrecomp.log";
    fs::rename(log, docs / "psxrecomp.prev.log", ec);
    /* One descriptor, append mode, shared by stdout and stderr (and by NSLog,
     * which writes to fd 2), so lines interleave in order instead of
     * overwriting each other. */
    {
        int fd = open(log.c_str(), O_WRONLY | O_CREAT | O_TRUNC | O_APPEND, 0644);
        if (fd >= 0) {
            std::fflush(stdout); std::fflush(stderr);
            dup2(fd, STDOUT_FILENO); dup2(fd, STDERR_FILENO);
            if (fd > STDERR_FILENO) close(fd);
            std::setvbuf(stdout, nullptr, _IOLBF, 0);
            std::setvbuf(stderr, nullptr, _IONBF, 0);
        }
    }

    const std::string real_argv0 = (*argc > 0 && (*argv)[0]) ? (*argv)[0] : "psx-runtime";
    const fs::path bundle = fs::path(real_argv0).parent_path();

    /* The runtime anchors every file it reads AND writes (settings, mods/state,
     * caches, texture packs) at the executable's folder, which on iOS is the
     * read-only bundle. Give it Documents as that folder instead: link the
     * bundle's read-only resources into it on every launch (the bundle path
     * changes with each install) and report the executable as living there. */
    static const char *const skip[] = {"Info.plist", "PkgInfo", "Frameworks", "_CodeSignature",
                                       "embedded.mobileprovision", "SC_Info", "mods"};
    const std::string exe_name = fs::path(real_argv0).filename().string();
    auto relink = [&](const fs::path &target, const fs::path &link) {
        std::error_code e;
        if (fs::is_symlink(link, e)) fs::remove(link, e);
        if (!fs::exists(link, e)) fs::create_symlink(target, link, e);
        if (e) std::fprintf(stderr, "psxrecomp-ios: link %s: %s\n", link.c_str(), e.message().c_str());
    };
    for (fs::directory_iterator it(bundle, ec), end; !ec && it != end; it.increment(ec)) {
        const std::string name = it->path().filename().string();
        if (name == exe_name) continue;
        bool skipped = false;
        for (const char *s : skip) skipped = skipped || name == s;
        if (!skipped) relink(it->path(), docs / name);
    }
    fs::create_directories(docs / "mods", ec);
    if (fs::exists(bundle / "mods" / "bundled", ec)) relink(bundle / "mods" / "bundled", docs / "mods" / "bundled");
    const std::string argv0 = (docs / exe_name).string();

    /* SDL's HIDAPI Steam Controller driver opens CoreBluetooth at joystick
     * init. Standard pads arrive through the Game Controller framework, so
     * leave that driver off (the Info.plist still declares Bluetooth use, as
     * iOS aborts an app that touches CoreBluetooth without it). */
    setenv("SDL_JOYSTICK_HIDAPI_STEAM", "0", 0);
    setenv("SDL_JOYSTICK_HIDAPI_STEAM_HORI", "0", 0);

    /* Caches (seamless-loading resources, resident packs) default to
     * $HOME/.cache, which the sandbox does not allow. */
    setenv("XDG_CACHE_HOME", (fs::path(home ? home : ".") / "Library" / "Caches").c_str(), 0);

    /* SDL loads Vulkan by name; point it at the MoltenVK the app carries. */
    const fs::path mvk = bundle / "Frameworks" / "MoltenVK.framework" / "MoltenVK";
    if (fs::exists(mvk, ec)) setenv("SDL_VULKAN_LIBRARY", mvk.c_str(), 0);

    std::vector<std::string> extra;
    {
        std::ifstream in(docs / "args.txt");
        for (std::string line; std::getline(in, line);) {
            while (!line.empty() && (line.back() == '\r' || line.back() == ' ')) line.pop_back();
            if (!line.empty() && line[0] != '#') extra.push_back(line);
        }
    }

    g_args.clear();
    g_args.push_back(argv0);
    g_args.push_back("--no-launcher");
    if (!has_flag(extra, "--game") && fs::exists(docs / "game.toml", ec)) {
        g_args.push_back("--game");
        g_args.push_back((docs / "game.toml").string());
    }
    if (!has_flag(extra, "--disc")) {
        std::vector<fs::path> cues;
        for (fs::directory_iterator it(docs, ec), end; !ec && it != end; it.increment(ec)) {
            std::string ext = it->path().extension().string();
            std::transform(ext.begin(), ext.end(), ext.begin(), ::tolower);
            if (ext == ".cue" || ext == ".chd") cues.push_back(it->path());
        }
        std::sort(cues.begin(), cues.end());
        if (!cues.empty()) {
            g_args.push_back("--disc");
            g_args.push_back(cues.front().string());
        } else {
            std::ofstream(docs / "PUT YOUR DISC IMAGE HERE.txt")
                << "Copy your own disc image (.cue and its .bin files, or a .chd) into this folder,\n"
                   "then start the app again.\n";
        }
    }
    if (!has_flag(extra, "--memcard-dir")) {
        g_args.push_back("--memcard-dir");
        g_args.push_back((docs / "saves").string());
    }
    if (!has_flag(extra, "--renderer")) {
        g_args.push_back("--renderer");
        g_args.push_back("vulkan");
    }
    for (const std::string &e : extra) g_args.push_back(e);

    g_argv.clear();
    for (std::string &s : g_args) g_argv.push_back(s.data());
    g_argv.push_back(nullptr);
    *argc = (int)g_args.size();
    *argv = g_argv.data();

    std::printf("psxrecomp-ios: bundle=%s\npsxrecomp-ios: documents=%s\npsxrecomp-ios: args:",
                bundle.c_str(), docs.c_str());
    for (int i = 1; i < *argc; ++i) std::printf(" [%s]", (*argv)[i]);
    std::printf("\n");
}

static std::atomic<int> g_ios_inactive{0};
static std::atomic<int> g_ios_watch{0};

static bool SDLCALL psx_ios_lifecycle_watch(void *, SDL_Event *e) {
    if (e->type == SDL_EVENT_WILL_ENTER_BACKGROUND || e->type == SDL_EVENT_DID_ENTER_BACKGROUND)
        g_ios_inactive.store(1);
    else if (e->type == SDL_EVENT_DID_ENTER_FOREGROUND)
        g_ios_inactive.store(0);
    return true;
}

extern "C" void psx_ios_wait_foreground(void) {
    if (!g_ios_watch.exchange(1)) SDL_AddEventWatch(psx_ios_lifecycle_watch, nullptr);
    if (!g_ios_inactive.load()) return;
    std::printf("psxrecomp-ios: app inactive, holding GPU work\n");
    while (g_ios_inactive.load()) {
        if (SDL_IsMainThread()) SDL_PumpEvents();
        SDL_Delay(30);
    }
    std::printf("psxrecomp-ios: app active again\n");
}
#endif
