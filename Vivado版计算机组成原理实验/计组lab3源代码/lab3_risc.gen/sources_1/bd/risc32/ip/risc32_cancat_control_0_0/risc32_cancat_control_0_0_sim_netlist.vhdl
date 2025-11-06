-- Copyright 1986-2022 Xilinx, Inc. All Rights Reserved.
-- --------------------------------------------------------------------------------
-- Tool Version: Vivado v.2022.2 (win64) Build 3671981 Fri Oct 14 05:00:03 MDT 2022
-- Date        : Mon Oct 28 16:04:53 2024
-- Host        : cop running 64-bit major release  (build 9200)
-- Command     : write_vhdl -force -mode funcsim
--               e:/project/lab3_risc/lab3_risc.gen/sources_1/bd/risc32/ip/risc32_cancat_control_0_0/risc32_cancat_control_0_0_sim_netlist.vhdl
-- Design      : risc32_cancat_control_0_0
-- Purpose     : This VHDL netlist is a functional simulation representation of the design and should not be modified or
--               synthesized. This netlist cannot be used for SDF annotated simulation.
-- Device      : xc7a100tcsg324-1
-- --------------------------------------------------------------------------------
library IEEE;
use IEEE.STD_LOGIC_1164.ALL;
library UNISIM;
use UNISIM.VCOMPONENTS.ALL;
entity risc32_cancat_control_0_0 is
  port (
    in0 : in STD_LOGIC_VECTOR ( 4 downto 0 );
    in1 : in STD_LOGIC_VECTOR ( 2 downto 0 );
    in2 : in STD_LOGIC;
    dout : out STD_LOGIC_VECTOR ( 8 downto 0 )
  );
  attribute NotValidForBitStream : boolean;
  attribute NotValidForBitStream of risc32_cancat_control_0_0 : entity is true;
  attribute CHECK_LICENSE_TYPE : string;
  attribute CHECK_LICENSE_TYPE of risc32_cancat_control_0_0 : entity is "risc32_cancat_control_0_0,cancat_control,{}";
  attribute DowngradeIPIdentifiedWarnings : string;
  attribute DowngradeIPIdentifiedWarnings of risc32_cancat_control_0_0 : entity is "yes";
  attribute IP_DEFINITION_SOURCE : string;
  attribute IP_DEFINITION_SOURCE of risc32_cancat_control_0_0 : entity is "module_ref";
  attribute X_CORE_INFO : string;
  attribute X_CORE_INFO of risc32_cancat_control_0_0 : entity is "cancat_control,Vivado 2022.2";
end risc32_cancat_control_0_0;

architecture STRUCTURE of risc32_cancat_control_0_0 is
  signal \^in0\ : STD_LOGIC_VECTOR ( 4 downto 0 );
  signal \^in1\ : STD_LOGIC_VECTOR ( 2 downto 0 );
  signal \^in2\ : STD_LOGIC;
begin
  \^in0\(4 downto 0) <= in0(4 downto 0);
  \^in1\(2 downto 0) <= in1(2 downto 0);
  \^in2\ <= in2;
  dout(8) <= \^in2\;
  dout(7 downto 5) <= \^in1\(2 downto 0);
  dout(4 downto 0) <= \^in0\(4 downto 0);
end STRUCTURE;
