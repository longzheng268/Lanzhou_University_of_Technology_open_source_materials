-- Copyright 1986-2022 Xilinx, Inc. All Rights Reserved.
-- --------------------------------------------------------------------------------
-- Tool Version: Vivado v.2022.2 (win64) Build 3671981 Fri Oct 14 05:00:03 MDT 2022
-- Date        : Mon Oct 28 16:04:12 2024
-- Host        : cop running 64-bit major release  (build 9200)
-- Command     : write_vhdl -force -mode funcsim
--               e:/project/lab3_risc/lab3_risc.gen/sources_1/bd/risc32/ip/risc32_slice_inst_0_0/risc32_slice_inst_0_0_sim_netlist.vhdl
-- Design      : risc32_slice_inst_0_0
-- Purpose     : This VHDL netlist is a functional simulation representation of the design and should not be modified or
--               synthesized. This netlist cannot be used for SDF annotated simulation.
-- Device      : xc7a100tcsg324-1
-- --------------------------------------------------------------------------------
library IEEE;
use IEEE.STD_LOGIC_1164.ALL;
library UNISIM;
use UNISIM.VCOMPONENTS.ALL;
entity risc32_slice_inst_0_0 is
  port (
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
  attribute NotValidForBitStream : boolean;
  attribute NotValidForBitStream of risc32_slice_inst_0_0 : entity is true;
  attribute CHECK_LICENSE_TYPE : string;
  attribute CHECK_LICENSE_TYPE of risc32_slice_inst_0_0 : entity is "risc32_slice_inst_0_0,slice_inst,{}";
  attribute DowngradeIPIdentifiedWarnings : string;
  attribute DowngradeIPIdentifiedWarnings of risc32_slice_inst_0_0 : entity is "yes";
  attribute IP_DEFINITION_SOURCE : string;
  attribute IP_DEFINITION_SOURCE of risc32_slice_inst_0_0 : entity is "module_ref";
  attribute X_CORE_INFO : string;
  attribute X_CORE_INFO of risc32_slice_inst_0_0 : entity is "slice_inst,Vivado 2022.2";
end risc32_slice_inst_0_0;

architecture STRUCTURE of risc32_slice_inst_0_0 is
  signal \^instruction\ : STD_LOGIC_VECTOR ( 31 downto 0 );
begin
  \^instruction\(31 downto 2) <= instruction(31 downto 2);
  inst_11_7(4 downto 0) <= \^instruction\(11 downto 7);
  inst_14_12(2 downto 0) <= \^instruction\(14 downto 12);
  inst_19_15(4 downto 0) <= \^instruction\(19 downto 15);
  inst_24_20(4 downto 0) <= \^instruction\(24 downto 20);
  inst_30 <= \^instruction\(30);
  inst_31 <= \^instruction\(31);
  inst_31_25(6 downto 0) <= \^instruction\(31 downto 25);
  inst_6_2(4 downto 0) <= \^instruction\(6 downto 2);
end STRUCTURE;
