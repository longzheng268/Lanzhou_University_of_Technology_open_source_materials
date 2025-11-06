-- Copyright 1986-2022 Xilinx, Inc. All Rights Reserved.
-- --------------------------------------------------------------------------------
-- Tool Version: Vivado v.2022.2 (win64) Build 3671981 Fri Oct 14 05:00:03 MDT 2022
-- Date        : Mon Oct 28 16:04:53 2024
-- Host        : cop running 64-bit major release  (build 9200)
-- Command     : write_vhdl -force -mode synth_stub
--               e:/project/lab3_risc/lab3_risc.gen/sources_1/bd/risc32/ip/risc32_cancat_control_0_0/risc32_cancat_control_0_0_stub.vhdl
-- Design      : risc32_cancat_control_0_0
-- Purpose     : Stub declaration of top-level module interface
-- Device      : xc7a100tcsg324-1
-- --------------------------------------------------------------------------------
library IEEE;
use IEEE.STD_LOGIC_1164.ALL;

entity risc32_cancat_control_0_0 is
  Port ( 
    in0 : in STD_LOGIC_VECTOR ( 4 downto 0 );
    in1 : in STD_LOGIC_VECTOR ( 2 downto 0 );
    in2 : in STD_LOGIC;
    dout : out STD_LOGIC_VECTOR ( 8 downto 0 )
  );

end risc32_cancat_control_0_0;

architecture stub of risc32_cancat_control_0_0 is
attribute syn_black_box : boolean;
attribute black_box_pad_pin : string;
attribute syn_black_box of stub : architecture is true;
attribute black_box_pad_pin of stub : architecture is "in0[4:0],in1[2:0],in2,dout[8:0]";
attribute X_CORE_INFO : string;
attribute X_CORE_INFO of stub : architecture is "cancat_control,Vivado 2022.2";
begin
end;
