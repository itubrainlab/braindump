brAIn stream for Windows 10 and 11 (64-bit)

Install for the current user (no administrator rights): double-click
install.cmd, then open brAIn stream from the Start menu. The first launch
downloads g.Pype and the other libraries from PyPI (several hundred MB, a
few minutes); later launches start directly. Or run it from this folder
without installing: "brAIn stream.cmd".

Devices:
- Unicorn: pair the headset in Settings > Bluetooth & devices first.
- Gazepoint GP3 or GP3 HD on this computer: install UsbDk once
  (https://github.com/daynix/UsbDk/releases, the x64 .msi). It leaves
  Gazepoint's driver in place, so Gazepoint Control keeps working (not at
  the same time). Or bind the WinUSB driver with Zadig
  (https://zadig.akeo.ie; GP3: 1E10 2001; GP3 HD: the composite device
  1E10 3300), which Gazepoint Control cannot use. An eye tracker on another
  computer needs nothing.
- Polar H10: wear the strap; it is found over Bluetooth LE.
- Shimmer3 GSR+: pair it in Settings > Bluetooth & devices (PIN 1234);
  choose its outgoing COM port in brAIn stream.

Remove it with uninstall.cmd; projects and recordings are kept.

brAIn stream is MIT licensed (LICENSE). It installs g.Pype and g.tec's
gtec-unicorn driver, under the g.tec Non-Commercial License (academic
research and teaching only), and pyshimmer, under the GPL-3.0.
