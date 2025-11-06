-- Copyright 1986-2022 Xilinx, Inc. All Rights Reserved.
-- --------------------------------------------------------------------------------
-- Tool Version: Vivado v.2022.2 (win64) Build 3671981 Fri Oct 14 05:00:03 MDT 2022
-- Date        : Tue Oct 29 10:56:53 2024
-- Host        : cop running 64-bit major release  (build 9200)
-- Command     : write_vhdl -force -mode funcsim
--               e:/project/lab3_risc/lab3_risc.gen/sources_1/bd/risc32/ip/risc32_control_rom_0_0/risc32_control_rom_0_0_sim_netlist.vhdl
-- Design      : risc32_control_rom_0_0
-- Purpose     : This VHDL netlist is a functional simulation representation of the design and should not be modified or
--               synthesized. This netlist cannot be used for SDF annotated simulation.
-- Device      : xc7a100tcsg324-1
-- --------------------------------------------------------------------------------
library IEEE;
use IEEE.STD_LOGIC_1164.ALL;
library UNISIM;
use UNISIM.VCOMPONENTS.ALL;
entity risc32_control_rom_0_0_control_rom is
  port (
    w_ena : out STD_LOGIC;
    bsel : out STD_LOGIC;
    mem_sel : out STD_LOGIC;
    imm_sel : out STD_LOGIC;
    inst_ctr : in STD_LOGIC_VECTOR ( 8 downto 0 )
  );
  attribute ORIG_REF_NAME : string;
  attribute ORIG_REF_NAME of risc32_control_rom_0_0_control_rom : entity is "control_rom";
end risc32_control_rom_0_0_control_rom;

architecture STRUCTURE of risc32_control_rom_0_0_control_rom is
  signal mem_ena_INST_0_i_1_n_0 : STD_LOGIC;
  attribute SOFT_HLUTNM : string;
  attribute SOFT_HLUTNM of bsel_INST_0 : label is "soft_lutpair0";
  attribute SOFT_HLUTNM of mem_ena_INST_0 : label is "soft_lutpair1";
  attribute SOFT_HLUTNM of mem_sel_INST_0 : label is "soft_lutpair1";
  attribute SOFT_HLUTNM of w_ena_INST_0 : label is "soft_lutpair0";
begin
bsel_INST_0: unisim.vcomponents.LUT5
    generic map(
      INIT => X"04003400"
    )
        port map (
      I0 => inst_ctr(8),
      I1 => inst_ctr(6),
      I2 => inst_ctr(2),
      I3 => mem_ena_INST_0_i_1_n_0,
      I4 => inst_ctr(3),
      O => bsel
    );
mem_ena_INST_0: unisim.vcomponents.LUT5
    generic map(
      INIT => X"00200000"
    )
        port map (
      I0 => inst_ctr(3),
      I1 => inst_ctr(2),
      I2 => mem_ena_INST_0_i_1_n_0,
      I3 => inst_ctr(8),
      I4 => inst_ctr(6),
      O => imm_sel
    );
mem_ena_INST_0_i_1: unisim.vcomponents.LUT5
    generic map(
      INIT => X"00000001"
    )
        port map (
      I0 => inst_ctr(0),
      I1 => inst_ctr(7),
      I2 => inst_ctr(4),
      I3 => inst_ctr(1),
      I4 => inst_ctr(5),
      O => mem_ena_INST_0_i_1_n_0
    );
mem_sel_INST_0: unisim.vcomponents.LUT5
    generic map(
      INIT => X"00100000"
    )
        port map (
      I0 => inst_ctr(3),
      I1 => inst_ctr(2),
      I2 => mem_ena_INST_0_i_1_n_0,
      I3 => inst_ctr(8),
      I4 => inst_ctr(6),
      O => mem_sel
    );
w_ena_INST_0: unisim.vcomponents.LUT5
    generic map(
      INIT => X"15000200"
    )
        port map (
      I0 => inst_ctr(6),
      I1 => inst_ctr(8),
      I2 => inst_ctr(3),
      I3 => mem_ena_INST_0_i_1_n_0,
      I4 => inst_ctr(2),
      O => w_ena
    );
end STRUCTURE;
library IEEE;
use IEEE.STD_LOGIC_1164.ALL;
library UNISIM;
use UNISIM.VCOMPONENTS.ALL;
entity risc32_control_rom_0_0 is
  port (
    inst_ctr : in STD_LOGIC_VECTOR ( 8 downto 0 );
    imm_sel : out STD_LOGIC;
    bsel : out STD_LOGIC;
    mem_sel : out STD_LOGIC;
    alusel : out STD_LOGIC_VECTOR ( 2 downto 0 );
    w_ena : out STD_LOGIC;
    mem_ena : out STD_LOGIC
  );
  attribute NotValidForBitStream : boolean;
  attribute NotValidForBitStream of risc32_control_rom_0_0 : entity is true;
  attribute CHECK_LICENSE_TYPE : string;
  attribute CHECK_LICENSE_TYPE of risc32_control_rom_0_0 : entity is "risc32_control_rom_0_0,control_rom,{}";
  attribute DowngradeIPIdentifiedWarnings : string;
  attribute DowngradeIPIdentifiedWarnings of risc32_control_rom_0_0 : entity is "yes";
  attribute IP_DEFINITION_SOURCE : string;
  attribute IP_DEFINITION_SOURCE of risc32_control_rom_0_0 : entity is "module_ref";
  attribute X_CORE_INFO : string;
  attribute X_CORE_INFO of risc32_control_rom_0_0 : entity is "control_rom,Vivado 2022.2";
end risc32_control_rom_0_0;

architecture STRUCTURE of risc32_control_rom_0_0 is
  signal \<const0>\ : STD_LOGIC;
  signal \^imm_sel\ : STD_LOGIC;
begin
  alusel(2) <= \<const0>\;
  alusel(1) <= \<const0>\;
  alusel(0) <= \<const0>\;
  imm_sel <= \^imm_sel\;
  mem_ena <= \^imm_sel\;
GND: unisim.vcomponents.GND
     port map (
      G => \<const0>\
    );
inst: entity work.risc32_control_rom_0_0_control_rom
     port map (
      bsel => bsel,
      imm_sel => \^imm_sel\,
      inst_ctr(8 downto 0) => inst_ctr(8 downto 0),
      mem_sel => mem_sel,
      w_ena => w_ena
    );
end STRUCTURE;
