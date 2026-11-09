brAIn dump for Windows 10 and 11 (64-bit)

Install for the current user (no administrator rights): double-click
install.cmd, then open brAIn dump from the Start menu. The first launch
downloads g.Pype and the other libraries from PyPI (several hundred MB, a
few minutes); later launches start directly. Or run it from this folder
without installing: "brAIn dump.cmd".

Devices:
- Unicorn: pair the headset in Settings > Bluetooth & devices first.
- Gazepoint GP3 on this computer: bind the WinUSB driver to the camera
  once with Zadig (https://zadig.akeo.ie, "USB 2.0 Digital Imaging
  Camera", 1E10 2001). Gazepoint Control cannot use the camera while it
  is bound. A GP3 on another computer needs nothing.
- Polar H10: wear the strap; it is found over Bluetooth LE.
- Shimmer3 GSR+: pair it in Settings > Bluetooth & devices (PIN 1234);
  choose its outgoing COM port in brAIn dump.

Remove it with uninstall.cmd; projects and recordings are kept.

brAIn dump is MIT licensed (LICENSE). It installs g.Pype and g.tec's
gtec-unicorn driver, under the g.tec Non-Commercial License (academic
research and teaching only), and pyshimmer, under the GPL-3.0.
