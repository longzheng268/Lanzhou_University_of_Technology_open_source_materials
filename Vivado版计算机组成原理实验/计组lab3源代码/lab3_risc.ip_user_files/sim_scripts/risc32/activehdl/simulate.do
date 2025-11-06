onbreak {quit -force}
onerror {quit -force}

asim +access +r +m+risc32  -L xpm -L xil_defaultlib -L xlconstant_v1_1_7 -L unisims_ver -L unimacro_ver -L secureip -O5 xil_defaultlib.risc32 xil_defaultlib.glbl

set NumericStdNoWarnings 1
set StdArithNoWarnings 1

do {wave.do}

view wave
view structure

do {risc32.udo}

run 1000ns

endsim

quit -force
