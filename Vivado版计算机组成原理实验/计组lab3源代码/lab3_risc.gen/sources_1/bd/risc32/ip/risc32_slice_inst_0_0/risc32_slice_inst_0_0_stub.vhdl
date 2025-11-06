-- Copyright 1986-2022 Xilinx, Inc. All Rights Reserved.
-- --------------------------------------------------------------------------------
-- Tool Version: Vivado v.2022.2 (win64) Build 3671981 Fri Oct 14 05:00:03 MDT 2022
-- Date        : Mon Oct 28 16:04:12 2024
-- Host        : cop running 64-bit major release  (build 9200)
-- Command     : write_vhdl -force -mode synth_stub
--               e:/project/lab3_risc/lab3_risc.gen/sources_1/bd/risc32/ip/risc32_slice_inst_0_0/risc32_slice_inst_0_0_stub.vhdl
-- Design      : risc32_slice_inst_0_0
-- Purpose     : Stub declaration of top-level module interface
-- Device      : xc7a100tcsg324-1
-- --------------------------------------------------------------------------------
library IEEE;
use IEEE.STD_LOGIC_1164.ALL;

entity risc32_slice_inst_0_0 is
  Port ( 
    instruction : in STD_LOGIC_VECTOR ( 31 downto 0 );
    inst_31 : out STD_LOGIC;
    inst_30 : out STD_LOGIC;
    inst_31_25 : out STD_LOGIC_VECTOR ( 6 downto 0 );
    inst_24_20 : out STD_LOGIC_VECTOR ( 4 downto 0 );
    inst_19_15 : out STD_LOGIC_VECTOR ( 4 downto 0 );
    inst_14_12 : out STD_LOGIC_VECTOR ( 2 downto 0 );
    inst_11_7 : out STD_LOGIC_VECTOR ( 4 downto 0 );
    inst_6_2 : out STD_LOGIC_VECTOR ( 4 downto 0 )
  );

end risc32_slice_inst_0_0;

architecture stub of risc32_slice_inst_0_0 is
attribute syn_black_box : boolean;
attribute black_box_pad_pin : string;
attribute syn_black_box of stub : architecture is true;
attribute black_box_pad_pin of stub : architecture is "instruction[31:0],inst_31,inst_30,inst_31_25[6:0],inst_24_20[4:0],inst_19_15[4:0],inst_14_12[2:0],inst_11_7[4:0],inst_6_2[4:0]";
attribute X_CORE_INFO : string;
attribute X_CORE_INFO of stub : architecture is "slice_inst,Vivado 2022.2";
begin
end;
