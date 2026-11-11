brAIn dump for Linux (x86_64, glibc 2.34+: Ubuntu 22.04, Debian 12,
Fedora 35 or later)

Install for the current user (no administrator rights):

    ./install.sh

then open brAIn dump from the applications menu, or run braindump. The
first launch downloads g.Pype and the other libraries from PyPI (several
hundred MB, a few minutes); later launches start directly. Or run it from
this folder without installing: ./braindump

Devices:
- Unicorn: switch the headset on; brAIn dump finds it over Bluetooth.
- Gazepoint GP3 or GP3 HD on this computer: the first time, OpenGP3 asks once for
  the administrator password to install a USB rule.
- Polar H10: wear the strap; it is found over Bluetooth (BlueZ).
- Shimmer3 GSR+: pair it (bluetoothctl: pair, trust; PIN 1234) and bind a
  serial port: sudo rfcomm bind 0 <address>. Your user needs to be in the
  dialout group.
- Qt needs the usual desktop libraries (on Debian/Ubuntu, libxcb-cursor0
  among them).

Remove it with ./uninstall.sh; projects and recordings are kept.

brAIn dump is MIT licensed (LICENSE). It installs g.Pype and g.tec's
gtec-unicorn driver, under the g.tec Non-Commercial License (academic
research and teaching only), and pyshimmer, under the GPL-3.0.
