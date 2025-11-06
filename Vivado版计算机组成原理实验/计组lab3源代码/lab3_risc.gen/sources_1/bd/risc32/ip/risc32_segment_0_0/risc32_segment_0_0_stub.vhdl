-- Copyright 1986-2022 Xilinx, Inc. All Rights Reserved.
-- --------------------------------------------------------------------------------
-- Tool Version: Vivado v.2022.2 (win64) Build 3671981 Fri Oct 14 05:00:03 MDT 2022
-- Date        : Tue Oct 29 10:56:54 2024
-- Host        : cop running 64-bit major release  (build 9200)
-- Command     : write_vhdl -force -mode synth_stub
--               e:/project/lab3_risc/lab3_risc.gen/sources_1/bd/risc32/ip/risc32_segment_0_0/risc32_segment_0_0_stub.vhdl
-- Design      : risc32_segment_0_0
-- Purpose     : Stub declaration of top-level module interface
-- Device      : xc7a100tcsg324-1
-- --------------------------------------------------------------------------------
library IEEE;
use IEEE.STD_LOGIC_1164.ALL;

entity risc32_segment_0_0 is
  Port ( 
    clk : in STD_LOGIC;
    rst_n : in STD_LOGIC;
    Data_i : in STD_LOGIC_VECTOR ( 31 downto 0 );
    AN : out STD_LOGIC_VECTOR ( 7 downto 0 );
    seg_data_o : out STD_LOGIC_VECTOR ( 7 downto 0 )
  );

end risc32_segment_0_0;

architecture stub of risc32_segment_0_0 is
attribute syn_black_box : boolean;
attribute black_box_pad_pin : string;
attribute syn_black_box of stub : architecture is true;
attribute black_box_pad_pin of stub : architecture is "clk,rst_n,Data_i[31:0],AN[7:0],seg_data_o[7:0]";
attribute X_CORE_INFO : string;
attribute X_CORE_INFO of stub : architecture is "segment,Vivado 2022.2";
begin
end;
