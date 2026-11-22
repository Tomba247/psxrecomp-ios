#!/bin/bash
# package_ipa.sh <built .app dir> <output .ipa> <display name> <bundle id> [MoltenVK.framework] [icon dir]
# The optional icon dir holds AppIcon60x60@2x.png, AppIcon60x60@3x.png and AppIcon76x76@2x~ipad.png.
# Finishes an iOS .app built by CMake/Ninja and packs it as an unsigned IPA.
# The archive is stored (uncompressed) with plain local headers, which
# on-device signers such as FlareStore require.
set -euo pipefail
APP="$1"; OUT="$2"; NAME="$3"; BID="$4"; MVK="${5:-}"; ICONS="${6:-}"
HERE="$(cd "$(dirname "$0")" && pwd)"
EXE="$(basename "$APP" .app)"
test -f "$APP/$EXE" || { echo "no executable $APP/$EXE"; ls -la "$APP"; exit 1; }
ICON_KEYS=""
if [ -n "$ICONS" ] && [ -f "$ICONS/AppIcon60x60@2x.png" ]; then
    cp "$ICONS"/AppIcon*.png "$APP/"
    ICON_KEYS='<key>CFBundleIcons</key><dict><key>CFBundlePrimaryIcon</key><dict><key>CFBundleIconFiles</key><array><string>AppIcon60x60</string></array><key>CFBundleIconName</key><string>AppIcon</string></dict></dict><key>CFBundleIcons~ipad</key><dict><key>CFBundlePrimaryIcon</key><dict><key>CFBundleIconFiles</key><array><string>AppIcon60x60</string><string>AppIcon76x76</string></array><key>CFBundleIconName</key><string>AppIcon</string></dict></dict>'
fi
sed -e "s/__EXECUTABLE__/$EXE/" -e "s/__BUNDLE_ID__/$BID/" -e "s/__NAME__/$NAME/g" -e "s|__ICON_KEYS__|$ICON_KEYS|" "$HERE/Info.plist" > "$APP/Info.plist"
plutil -lint "$APP/Info.plist"
printf 'APPL????' > "$APP/PkgInfo"
if [ -n "$MVK" ]; then
    mkdir -p "$APP/Frameworks"
    rm -rf "$APP/Frameworks/MoltenVK.framework"
    cp -a "$MVK" "$APP/Frameworks/"
fi
WORK="$(mktemp -d)"; mkdir "$WORK/Payload"; cp -a "$APP" "$WORK/Payload/"
rm -f "$OUT"
(cd "$WORK" && zip -q -r -X -y -0 "$OUT" Payload)
rm -rf "$WORK"
unzip -l "$OUT" | tail -1
find "$APP" -maxdepth 2 | head -40
ls -la "$OUT"
