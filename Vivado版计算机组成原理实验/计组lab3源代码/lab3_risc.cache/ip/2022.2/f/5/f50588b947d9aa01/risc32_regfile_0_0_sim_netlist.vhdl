-- Copyright 1986-2022 Xilinx, Inc. All Rights Reserved.
-- --------------------------------------------------------------------------------
-- Tool Version: Vivado v.2022.2 (win64) Build 3671981 Fri Oct 14 05:00:03 MDT 2022
-- Date        : Mon Oct 28 16:04:11 2024
-- Host        : cop running 64-bit major release  (build 9200)
-- Command     : write_vhdl -force -mode funcsim -rename_top decalper_eb_ot_sdeen_pot_pi_dehcac_xnilix -prefix
--               decalper_eb_ot_sdeen_pot_pi_dehcac_xnilix_ risc32_regfile_0_0_sim_netlist.vhdl
-- Design      : risc32_regfile_0_0
-- Purpose     : This VHDL netlist is a functional simulation representation of the design and should not be modified or
--               synthesized. This netlist cannot be used for SDF annotated simulation.
-- Device      : xc7a100tcsg324-1
-- --------------------------------------------------------------------------------
library IEEE;
use IEEE.STD_LOGIC_1164.ALL;
library UNISIM;
use UNISIM.VCOMPONENTS.ALL;
entity decalper_eb_ot_sdeen_pot_pi_dehcac_xnilix_regfile is
  port (
    reg_data1 : out STD_LOGIC_VECTOR ( 31 downto 0 );
    reg_data2 : out STD_LOGIC_VECTOR ( 31 downto 0 );
    clk : in STD_LOGIC;
    w_data : in STD_LOGIC_VECTOR ( 31 downto 0 );
    w_ena : in STD_LOGIC;
    w_addr : in STD_LOGIC_VECTOR ( 4 downto 0 );
    r_addr1 : in STD_LOGIC_VECTOR ( 4 downto 0 );
    r_addr2 : in STD_LOGIC_VECTOR ( 4 downto 0 )
  );
end decalper_eb_ot_sdeen_pot_pi_dehcac_xnilix_regfile;

architecture STRUCTURE of decalper_eb_ot_sdeen_pot_pi_dehcac_xnilix_regfile is
  signal reg_data10 : STD_LOGIC_VECTOR ( 31 downto 0 );
  signal reg_data20 : STD_LOGIC_VECTOR ( 31 downto 0 );
  signal NLW_reg_array_reg_r1_0_31_0_5_DOD_UNCONNECTED : STD_LOGIC_VECTOR ( 1 downto 0 );
  signal NLW_reg_array_reg_r1_0_31_12_17_DOD_UNCONNECTED : STD_LOGIC_VECTOR ( 1 downto 0 );
  signal NLW_reg_array_reg_r1_0_31_18_23_DOD_UNCONNECTED : STD_LOGIC_VECTOR ( 1 downto 0 );
  signal NLW_reg_array_reg_r1_0_31_24_29_DOD_UNCONNECTED : STD_LOGIC_VECTOR ( 1 downto 0 );
  signal NLW_reg_array_reg_r1_0_31_30_31_SPO_UNCONNECTED : STD_LOGIC;
  signal \NLW_reg_array_reg_r1_0_31_30_31__0_SPO_UNCONNECTED\ : STD_LOGIC;
  signal NLW_reg_array_reg_r1_0_31_6_11_DOD_UNCONNECTED : STD_LOGIC_VECTOR ( 1 downto 0 );
  signal NLW_reg_array_reg_r2_0_31_0_5_DOD_UNCONNECTED : STD_LOGIC_VECTOR ( 1 downto 0 );
  signal NLW_reg_array_reg_r2_0_31_12_17_DOD_UNCONNECTED : STD_LOGIC_VECTOR ( 1 downto 0 );
  signal NLW_reg_array_reg_r2_0_31_18_23_DOD_UNCONNECTED : STD_LOGIC_VECTOR ( 1 downto 0 );
  signal NLW_reg_array_reg_r2_0_31_24_29_DOD_UNCONNECTED : STD_LOGIC_VECTOR ( 1 downto 0 );
  signal NLW_reg_array_reg_r2_0_31_30_31_SPO_UNCONNECTED : STD_LOGIC;
  signal \NLW_reg_array_reg_r2_0_31_30_31__0_SPO_UNCONNECTED\ : STD_LOGIC;
  signal NLW_reg_array_reg_r2_0_31_6_11_DOD_UNCONNECTED : STD_LOGIC_VECTOR ( 1 downto 0 );
  attribute METHODOLOGY_DRC_VIOS : string;
  attribute METHODOLOGY_DRC_VIOS of reg_array_reg_r1_0_31_0_5 : label is "";
  attribute RTL_RAM_BITS : integer;
  attribute RTL_RAM_BITS of reg_array_reg_r1_0_31_0_5 : label is 1024;
  attribute RTL_RAM_NAME : string;
  attribute RTL_RAM_NAME of reg_array_reg_r1_0_31_0_5 : label is "inst/reg_array_reg_r1_0_31_0_5";
  attribute RTL_RAM_TYPE : string;
  attribute RTL_RAM_TYPE of reg_array_reg_r1_0_31_0_5 : label is "RAM_SDP";
  attribute ram_addr_begin : integer;
  attribute ram_addr_begin of reg_array_reg_r1_0_31_0_5 : label is 0;
  attribute ram_addr_end : integer;
  attribute ram_addr_end of reg_array_reg_r1_0_31_0_5 : label is 31;
  attribute ram_offset : integer;
  attribute ram_offset of reg_array_reg_r1_0_31_0_5 : label is 0;
  attribute ram_slice_begin : integer;
  attribute ram_slice_begin of reg_array_reg_r1_0_31_0_5 : label is 0;
  attribute ram_slice_end : integer;
  attribute ram_slice_end of reg_array_reg_r1_0_31_0_5 : label is 5;
  attribute METHODOLOGY_DRC_VIOS of reg_array_reg_r1_0_31_12_17 : label is "";
  attribute RTL_RAM_BITS of reg_array_reg_r1_0_31_12_17 : label is 1024;
  attribute RTL_RAM_NAME of reg_array_reg_r1_0_31_12_17 : label is "inst/reg_array_reg_r1_0_31_12_17";
  attribute RTL_RAM_TYPE of reg_array_reg_r1_0_31_12_17 : label is "RAM_SDP";
  attribute ram_addr_begin of reg_array_reg_r1_0_31_12_17 : label is 0;
  attribute ram_addr_end of reg_array_reg_r1_0_31_12_17 : label is 31;
  attribute ram_offset of reg_array_reg_r1_0_31_12_17 : label is 0;
  attribute ram_slice_begin of reg_array_reg_r1_0_31_12_17 : label is 12;
  attribute ram_slice_end of reg_array_reg_r1_0_31_12_17 : label is 17;
  attribute METHODOLOGY_DRC_VIOS of reg_array_reg_r1_0_31_18_23 : label is "";
  attribute RTL_RAM_BITS of reg_array_reg_r1_0_31_18_23 : label is 1024;
  attribute RTL_RAM_NAME of reg_array_reg_r1_0_31_18_23 : label is "inst/reg_array_reg_r1_0_31_18_23";
  attribute RTL_RAM_TYPE of reg_array_reg_r1_0_31_18_23 : label is "RAM_SDP";
  attribute ram_addr_begin of reg_array_reg_r1_0_31_18_23 : label is 0;
  attribute ram_addr_end of reg_array_reg_r1_0_31_18_23 : label is 31;
  attribute ram_offset of reg_array_reg_r1_0_31_18_23 : label is 0;
  attribute ram_slice_begin of reg_array_reg_r1_0_31_18_23 : label is 18;
  attribute ram_slice_end of reg_array_reg_r1_0_31_18_23 : label is 23;
  attribute METHODOLOGY_DRC_VIOS of reg_array_reg_r1_0_31_24_29 : label is "";
  attribute RTL_RAM_BITS of reg_array_reg_r1_0_31_24_29 : label is 1024;
  attribute RTL_RAM_NAME of reg_array_reg_r1_0_31_24_29 : label is "inst/reg_array_reg_r1_0_31_24_29";
  attribute RTL_RAM_TYPE of reg_array_reg_r1_0_31_24_29 : label is "RAM_SDP";
  attribute ram_addr_begin of reg_array_reg_r1_0_31_24_29 : label is 0;
  attribute ram_addr_end of reg_array_reg_r1_0_31_24_29 : label is 31;
  attribute ram_offset of reg_array_reg_r1_0_31_24_29 : label is 0;
  attribute ram_slice_begin of reg_array_reg_r1_0_31_24_29 : label is 24;
  attribute ram_slice_end of reg_array_reg_r1_0_31_24_29 : label is 29;
  attribute METHODOLOGY_DRC_VIOS of reg_array_reg_r1_0_31_30_31 : label is "";
  attribute RTL_RAM_BITS of reg_array_reg_r1_0_31_30_31 : label is 1024;
  attribute RTL_RAM_NAME of reg_array_reg_r1_0_31_30_31 : label is "inst/reg_array_reg_r1_0_31_30_31";
  attribute RTL_RAM_TYPE of reg_array_reg_r1_0_31_30_31 : label is "RAM_SDP";
  attribute ram_addr_begin of reg_array_reg_r1_0_31_30_31 : label is 0;
  attribute ram_addr_end of reg_array_reg_r1_0_31_30_31 : label is 31;
  attribute ram_offset of reg_array_reg_r1_0_31_30_31 : label is 0;
  attribute ram_slice_begin of reg_array_reg_r1_0_31_30_31 : label is 30;
  attribute ram_slice_end of reg_array_reg_r1_0_31_30_31 : label is 31;
  attribute METHODOLOGY_DRC_VIOS of \reg_array_reg_r1_0_31_30_31__0\ : label is "";
  attribute RTL_RAM_BITS of \reg_array_reg_r1_0_31_30_31__0\ : label is 1024;
  attribute RTL_RAM_NAME of \reg_array_reg_r1_0_31_30_31__0\ : label is "inst/reg_array_reg_r1_0_31_30_31";
  attribute RTL_RAM_TYPE of \reg_array_reg_r1_0_31_30_31__0\ : label is "RAM_SDP";
  attribute ram_addr_begin of \reg_array_reg_r1_0_31_30_31__0\ : label is 0;
  attribute ram_addr_end of \reg_array_reg_r1_0_31_30_31__0\ : label is 31;
  attribute ram_offset of \reg_array_reg_r1_0_31_30_31__0\ : label is 0;
  attribute ram_slice_begin of \reg_array_reg_r1_0_31_30_31__0\ : label is 30;
  attribute ram_slice_end of \reg_array_reg_r1_0_31_30_31__0\ : label is 31;
  attribute METHODOLOGY_DRC_VIOS of reg_array_reg_r1_0_31_6_11 : label is "";
  attribute RTL_RAM_BITS of reg_array_reg_r1_0_31_6_11 : label is 1024;
  attribute RTL_RAM_NAME of reg_array_reg_r1_0_31_6_11 : label is "inst/reg_array_reg_r1_0_31_6_11";
  attribute RTL_RAM_TYPE of reg_array_reg_r1_0_31_6_11 : label is "RAM_SDP";
  attribute ram_addr_begin of reg_array_reg_r1_0_31_6_11 : label is 0;
  attribute ram_addr_end of reg_array_reg_r1_0_31_6_11 : label is 31;
  attribute ram_offset of reg_array_reg_r1_0_31_6_11 : label is 0;
  attribute ram_slice_begin of reg_array_reg_r1_0_31_6_11 : label is 6;
  attribute ram_slice_end of reg_array_reg_r1_0_31_6_11 : label is 11;
  attribute METHODOLOGY_DRC_VIOS of reg_array_reg_r2_0_31_0_5 : label is "";
  attribute RTL_RAM_BITS of reg_array_reg_r2_0_31_0_5 : label is 1024;
  attribute RTL_RAM_NAME of reg_array_reg_r2_0_31_0_5 : label is "inst/reg_array_reg_r2_0_31_0_5";
  attribute RTL_RAM_TYPE of reg_array_reg_r2_0_31_0_5 : label is "RAM_SDP";
  attribute ram_addr_begin of reg_array_reg_r2_0_31_0_5 : label is 0;
  attribute ram_addr_end of reg_array_reg_r2_0_31_0_5 : label is 31;
  attribute ram_offset of reg_array_reg_r2_0_31_0_5 : label is 0;
  attribute ram_slice_begin of reg_array_reg_r2_0_31_0_5 : label is 0;
  attribute ram_slice_end of reg_array_reg_r2_0_31_0_5 : label is 5;
  attribute METHODOLOGY_DRC_VIOS of reg_array_reg_r2_0_31_12_17 : label is "";
  attribute RTL_RAM_BITS of reg_array_reg_r2_0_31_12_17 : label is 1024;
  attribute RTL_RAM_NAME of reg_array_reg_r2_0_31_12_17 : label is "inst/reg_array_reg_r2_0_31_12_17";
  attribute RTL_RAM_TYPE of reg_array_reg_r2_0_31_12_17 : label is "RAM_SDP";
  attribute ram_addr_begin of reg_array_reg_r2_0_31_12_17 : label is 0;
  attribute ram_addr_end of reg_array_reg_r2_0_31_12_17 : label is 31;
  attribute ram_offset of reg_array_reg_r2_0_31_12_17 : label is 0;
  attribute ram_slice_begin of reg_array_reg_r2_0_31_12_17 : label is 12;
  attribute ram_slice_end of reg_array_reg_r2_0_31_12_17 : label is 17;
  attribute METHODOLOGY_DRC_VIOS of reg_array_reg_r2_0_31_18_23 : label is "";
  attribute RTL_RAM_BITS of reg_array_reg_r2_0_31_18_23 : label is 1024;
  attribute RTL_RAM_NAME of reg_array_reg_r2_0_31_18_23 : label is "inst/reg_array_reg_r2_0_31_18_23";
  attribute RTL_RAM_TYPE of reg_array_reg_r2_0_31_18_23 : label is "RAM_SDP";
  attribute ram_addr_begin of reg_array_reg_r2_0_31_18_23 : label is 0;
  attribute ram_addr_end of reg_array_reg_r2_0_31_18_23 : label is 31;
  attribute ram_offset of reg_array_reg_r2_0_31_18_23 : label is 0;
  attribute ram_slice_begin of reg_array_reg_r2_0_31_18_23 : label is 18;
  attribute ram_slice_end of reg_array_reg_r2_0_31_18_23 : label is 23;
  attribute METHODOLOGY_DRC_VIOS of reg_array_reg_r2_0_31_24_29 : label is "";
  attribute RTL_RAM_BITS of reg_array_reg_r2_0_31_24_29 : label is 1024;
  attribute RTL_RAM_NAME of reg_array_reg_r2_0_31_24_29 : label is "inst/reg_array_reg_r2_0_31_24_29";
  attribute RTL_RAM_TYPE of reg_array_reg_r2_0_31_24_29 : label is "RAM_SDP";
  attribute ram_addr_begin of reg_array_reg_r2_0_31_24_29 : label is 0;
  attribute ram_addr_end of reg_array_reg_r2_0_31_24_29 : label is 31;
  attribute ram_offset of reg_array_reg_r2_0_31_24_29 : label is 0;
  attribute ram_slice_begin of reg_array_reg_r2_0_31_24_29 : label is 24;
  attribute ram_slice_end of reg_array_reg_r2_0_31_24_29 : label is 29;
  attribute METHODOLOGY_DRC_VIOS of reg_array_reg_r2_0_31_30_31 : label is "";
  attribute RTL_RAM_BITS of reg_array_reg_r2_0_31_30_31 : label is 1024;
  attribute RTL_RAM_NAME of reg_array_reg_r2_0_31_30_31 : label is "inst/reg_array_reg_r2_0_31_30_31";
  attribute RTL_RAM_TYPE of reg_array_reg_r2_0_31_30_31 : label is "RAM_SDP";
  attribute ram_addr_begin of reg_array_reg_r2_0_31_30_31 : label is 0;
  attribute ram_addr_end of reg_array_reg_r2_0_31_30_31 : label is 31;
  attribute ram_offset of reg_array_reg_r2_0_31_30_31 : label is 0;
  attribute ram_slice_begin of reg_array_reg_r2_0_31_30_31 : label is 30;
  attribute ram_slice_end of reg_array_reg_r2_0_31_30_31 : label is 31;
  attribute METHODOLOGY_DRC_VIOS of \reg_array_reg_r2_0_31_30_31__0\ : label is "";
  attribute RTL_RAM_BITS of \reg_array_reg_r2_0_31_30_31__0\ : label is 1024;
  attribute RTL_RAM_NAME of \reg_array_reg_r2_0_31_30_31__0\ : label is "inst/reg_array_reg_r2_0_31_30_31";
  attribute RTL_RAM_TYPE of \reg_array_reg_r2_0_31_30_31__0\ : label is "RAM_SDP";
  attribute ram_addr_begin of \reg_array_reg_r2_0_31_30_31__0\ : label is 0;
  attribute ram_addr_end of \reg_array_reg_r2_0_31_30_31__0\ : label is 31;
  attribute ram_offset of \reg_array_reg_r2_0_31_30_31__0\ : label is 0;
  attribute ram_slice_begin of \reg_array_reg_r2_0_31_30_31__0\ : label is 30;
  attribute ram_slice_end of \reg_array_reg_r2_0_31_30_31__0\ : label is 31;
  attribute METHODOLOGY_DRC_VIOS of reg_array_reg_r2_0_31_6_11 : label is "";
  attribute RTL_RAM_BITS of reg_array_reg_r2_0_31_6_11 : label is 1024;
  attribute RTL_RAM_NAME of reg_array_reg_r2_0_31_6_11 : label is "inst/reg_array_reg_r2_0_31_6_11";
  attribute RTL_RAM_TYPE of reg_array_reg_r2_0_31_6_11 : label is "RAM_SDP";
  attribute ram_addr_begin of reg_array_reg_r2_0_31_6_11 : label is 0;
  attribute ram_addr_end of reg_array_reg_r2_0_31_6_11 : label is 31;
  attribute ram_offset of reg_array_reg_r2_0_31_6_11 : label is 0;
  attribute ram_slice_begin of reg_array_reg_r2_0_31_6_11 : label is 6;
  attribute ram_slice_end of reg_array_reg_r2_0_31_6_11 : label is 11;
begin
reg_array_reg_r1_0_31_0_5: unisim.vcomponents.RAM32M
     port map (
      ADDRA(4 downto 0) => r_addr1(4 downto 0),
      ADDRB(4 downto 0) => r_addr1(4 downto 0),
      ADDRC(4 downto 0) => r_addr1(4 downto 0),
      ADDRD(4 downto 0) => w_addr(4 downto 0),
      DIA(1 downto 0) => w_data(1 downto 0),
      DIB(1 downto 0) => w_data(3 downto 2),
      DIC(1 downto 0) => w_data(5 downto 4),
      DID(1 downto 0) => B"00",
      DOA(1 downto 0) => reg_data10(1 downto 0),
      DOB(1 downto 0) => reg_data10(3 downto 2),
      DOC(1 downto 0) => reg_data10(5 downto 4),
      DOD(1 downto 0) => NLW_reg_array_reg_r1_0_31_0_5_DOD_UNCONNECTED(1 downto 0),
      WCLK => clk,
      WE => w_ena
    );
reg_array_reg_r1_0_31_12_17: unisim.vcomponents.RAM32M
     port map (
      ADDRA(4 downto 0) => r_addr1(4 downto 0),
      ADDRB(4 downto 0) => r_addr1(4 downto 0),
      ADDRC(4 downto 0) => r_addr1(4 downto 0),
      ADDRD(4 downto 0) => w_addr(4 downto 0),
      DIA(1 downto 0) => w_data(13 downto 12),
      DIB(1 downto 0) => w_data(15 downto 14),
      DIC(1 downto 0) => w_data(17 downto 16),
      DID(1 downto 0) => B"00",
      DOA(1 downto 0) => reg_data10(13 downto 12),
      DOB(1 downto 0) => reg_data10(15 downto 14),
      DOC(1 downto 0) => reg_data10(17 downto 16),
      DOD(1 downto 0) => NLW_reg_array_reg_r1_0_31_12_17_DOD_UNCONNECTED(1 downto 0),
      WCLK => clk,
      WE => w_ena
    );
reg_array_reg_r1_0_31_18_23: unisim.vcomponents.RAM32M
     port map (
      ADDRA(4 downto 0) => r_addr1(4 downto 0),
      ADDRB(4 downto 0) => r_addr1(4 downto 0),
      ADDRC(4 downto 0) => r_addr1(4 downto 0),
      ADDRD(4 downto 0) => w_addr(4 downto 0),
      DIA(1 downto 0) => w_data(19 downto 18),
      DIB(1 downto 0) => w_data(21 downto 20),
      DIC(1 downto 0) => w_data(23 downto 22),
      DID(1 downto 0) => B"00",
      DOA(1 downto 0) => reg_data10(19 downto 18),
      DOB(1 downto 0) => reg_data10(21 downto 20),
      DOC(1 downto 0) => reg_data10(23 downto 22),
      DOD(1 downto 0) => NLW_reg_array_reg_r1_0_31_18_23_DOD_UNCONNECTED(1 downto 0),
      WCLK => clk,
      WE => w_ena
    );
reg_array_reg_r1_0_31_24_29: unisim.vcomponents.RAM32M
     port map (
      ADDRA(4 downto 0) => r_addr1(4 downto 0),
      ADDRB(4 downto 0) => r_addr1(4 downto 0),
      ADDRC(4 downto 0) => r_addr1(4 downto 0),
      ADDRD(4 downto 0) => w_addr(4 downto 0),
      DIA(1 downto 0) => w_data(25 downto 24),
      DIB(1 downto 0) => w_data(27 downto 26),
      DIC(1 downto 0) => w_data(29 downto 28),
      DID(1 downto 0) => B"00",
      DOA(1 downto 0) => reg_data10(25 downto 24),
      DOB(1 downto 0) => reg_data10(27 downto 26),
      DOC(1 downto 0) => reg_data10(29 downto 28),
      DOD(1 downto 0) => NLW_reg_array_reg_r1_0_31_24_29_DOD_UNCONNECTED(1 downto 0),
      WCLK => clk,
      WE => w_ena
    );
reg_array_reg_r1_0_31_30_31: unisim.vcomponents.RAM32X1D
     port map (
      A0 => w_addr(0),
      A1 => w_addr(1),
      A2 => w_addr(2),
      A3 => w_addr(3),
      A4 => w_addr(4),
      D => w_data(30),
      DPO => reg_data10(30),
      DPRA0 => r_addr1(0),
      DPRA1 => r_addr1(1),
      DPRA2 => r_addr1(2),
      DPRA3 => r_addr1(3),
      DPRA4 => r_addr1(4),
      SPO => NLW_reg_array_reg_r1_0_31_30_31_SPO_UNCONNECTED,
      WCLK => clk,
      WE => w_ena
    );
\reg_array_reg_r1_0_31_30_31__0\: unisim.vcomponents.RAM32X1D
     port map (
      A0 => w_addr(0),
      A1 => w_addr(1),
      A2 => w_addr(2),
      A3 => w_addr(3),
      A4 => w_addr(4),
      D => w_data(31),
      DPO => reg_data10(31),
      DPRA0 => r_addr1(0),
      DPRA1 => r_addr1(1),
      DPRA2 => r_addr1(2),
      DPRA3 => r_addr1(3),
      DPRA4 => r_addr1(4),
      SPO => \NLW_reg_array_reg_r1_0_31_30_31__0_SPO_UNCONNECTED\,
      WCLK => clk,
      WE => w_ena
    );
reg_array_reg_r1_0_31_6_11: unisim.vcomponents.RAM32M
     port map (
      ADDRA(4 downto 0) => r_addr1(4 downto 0),
      ADDRB(4 downto 0) => r_addr1(4 downto 0),
      ADDRC(4 downto 0) => r_addr1(4 downto 0),
      ADDRD(4 downto 0) => w_addr(4 downto 0),
      DIA(1 downto 0) => w_data(7 downto 6),
      DIB(1 downto 0) => w_data(9 downto 8),
      DIC(1 downto 0) => w_data(11 downto 10),
      DID(1 downto 0) => B"00",
      DOA(1 downto 0) => reg_data10(7 downto 6),
      DOB(1 downto 0) => reg_data10(9 downto 8),
      DOC(1 downto 0) => reg_data10(11 downto 10),
      DOD(1 downto 0) => NLW_reg_array_reg_r1_0_31_6_11_DOD_UNCONNECTED(1 downto 0),
      WCLK => clk,
      WE => w_ena
    );
reg_array_reg_r2_0_31_0_5: unisim.vcomponents.RAM32M
     port map (
      ADDRA(4 downto 0) => r_addr2(4 downto 0),
      ADDRB(4 downto 0) => r_addr2(4 downto 0),
      ADDRC(4 downto 0) => r_addr2(4 downto 0),
      ADDRD(4 downto 0) => w_addr(4 downto 0),
      DIA(1 downto 0) => w_data(1 downto 0),
      DIB(1 downto 0) => w_data(3 downto 2),
      DIC(1 downto 0) => w_data(5 downto 4),
      DID(1 downto 0) => B"00",
      DOA(1 downto 0) => reg_data20(1 downto 0),
      DOB(1 downto 0) => reg_data20(3 downto 2),
      DOC(1 downto 0) => reg_data20(5 downto 4),
      DOD(1 downto 0) => NLW_reg_array_reg_r2_0_31_0_5_DOD_UNCONNECTED(1 downto 0),
      WCLK => clk,
      WE => w_ena
    );
reg_array_reg_r2_0_31_12_17: unisim.vcomponents.RAM32M
     port map (
      ADDRA(4 downto 0) => r_addr2(4 downto 0),
      ADDRB(4 downto 0) => r_addr2(4 downto 0),
      ADDRC(4 downto 0) => r_addr2(4 downto 0),
      ADDRD(4 downto 0) => w_addr(4 downto 0),
      DIA(1 downto 0) => w_data(13 downto 12),
      DIB(1 downto 0) => w_data(15 downto 14),
      DIC(1 downto 0) => w_data(17 downto 16),
      DID(1 downto 0) => B"00",
      DOA(1 downto 0) => reg_data20(13 downto 12),
      DOB(1 downto 0) => reg_data20(15 downto 14),
      DOC(1 downto 0) => reg_data20(17 downto 16),
      DOD(1 downto 0) => NLW_reg_array_reg_r2_0_31_12_17_DOD_UNCONNECTED(1 downto 0),
      WCLK => clk,
      WE => w_ena
    );
reg_array_reg_r2_0_31_18_23: unisim.vcomponents.RAM32M
     port map (
      ADDRA(4 downto 0) => r_addr2(4 downto 0),
      ADDRB(4 downto 0) => r_addr2(4 downto 0),
      ADDRC(4 downto 0) => r_addr2(4 downto 0),
      ADDRD(4 downto 0) => w_addr(4 downto 0),
      DIA(1 downto 0) => w_data(19 downto 18),
      DIB(1 downto 0) => w_data(21 downto 20),
      DIC(1 downto 0) => w_data(23 downto 22),
      DID(1 downto 0) => B"00",
      DOA(1 downto 0) => reg_data20(19 downto 18),
      DOB(1 downto 0) => reg_data20(21 downto 20),
      DOC(1 downto 0) => reg_data20(23 downto 22),
      DOD(1 downto 0) => NLW_reg_array_reg_r2_0_31_18_23_DOD_UNCONNECTED(1 downto 0),
      WCLK => clk,
      WE => w_ena
    );
reg_array_reg_r2_0_31_24_29: unisim.vcomponents.RAM32M
     port map (
      ADDRA(4 downto 0) => r_addr2(4 downto 0),
      ADDRB(4 downto 0) => r_addr2(4 downto 0),
      ADDRC(4 downto 0) => r_addr2(4 downto 0),
      ADDRD(4 downto 0) => w_addr(4 downto 0),
      DIA(1 downto 0) => w_data(25 downto 24),
      DIB(1 downto 0) => w_data(27 downto 26),
      DIC(1 downto 0) => w_data(29 downto 28),
      DID(1 downto 0) => B"00",
      DOA(1 downto 0) => reg_data20(25 downto 24),
      DOB(1 downto 0) => reg_data20(27 downto 26),
      DOC(1 downto 0) => reg_data20(29 downto 28),
      DOD(1 downto 0) => NLW_reg_array_reg_r2_0_31_24_29_DOD_UNCONNECTED(1 downto 0),
      WCLK => clk,
      WE => w_ena
    );
reg_array_reg_r2_0_31_30_31: unisim.vcomponents.RAM32X1D
     port map (
      A0 => w_addr(0),
      A1 => w_addr(1),
      A2 => w_addr(2),
      A3 => w_addr(3),
      A4 => w_addr(4),
      D => w_data(30),
      DPO => reg_data20(30),
      DPRA0 => r_addr2(0),
      DPRA1 => r_addr2(1),
      DPRA2 => r_addr2(2),
      DPRA3 => r_addr2(3),
      DPRA4 => r_addr2(4),
      SPO => NLW_reg_array_reg_r2_0_31_30_31_SPO_UNCONNECTED,
      WCLK => clk,
      WE => w_ena
    );
\reg_array_reg_r2_0_31_30_31__0\: unisim.vcomponents.RAM32X1D
     port map (
      A0 => w_addr(0),
      A1 => w_addr(1),
      A2 => w_addr(2),
      A3 => w_addr(3),
      A4 => w_addr(4),
      D => w_data(31),
      DPO => reg_data20(31),
      DPRA0 => r_addr2(0),
      DPRA1 => r_addr2(1),
      DPRA2 => r_addr2(2),
      DPRA3 => r_addr2(3),
      DPRA4 => r_addr2(4),
      SPO => \NLW_reg_array_reg_r2_0_31_30_31__0_SPO_UNCONNECTED\,
      WCLK => clk,
      WE => w_ena
    );
reg_array_reg_r2_0_31_6_11: unisim.vcomponents.RAM32M
     port map (
      ADDRA(4 downto 0) => r_addr2(4 downto 0),
      ADDRB(4 downto 0) => r_addr2(4 downto 0),
      ADDRC(4 downto 0) => r_addr2(4 downto 0),
      ADDRD(4 downto 0) => w_addr(4 downto 0),
      DIA(1 downto 0) => w_data(7 downto 6),
      DIB(1 downto 0) => w_data(9 downto 8),
      DIC(1 downto 0) => w_data(11 downto 10),
      DID(1 downto 0) => B"00",
      DOA(1 downto 0) => reg_data20(7 downto 6),
      DOB(1 downto 0) => reg_data20(9 downto 8),
      DOC(1 downto 0) => reg_data20(11 downto 10),
      DOD(1 downto 0) => NLW_reg_array_reg_r2_0_31_6_11_DOD_UNCONNECTED(1 downto 0),
      WCLK => clk,
      WE => w_ena
    );
\reg_data1[0]_INST_0\: unisim.vcomponents.LUT6
    generic map(
      INIT => X"FFFFFFFE00000000"
    )
        port map (
      I0 => r_addr1(4),
      I1 => r_addr1(3),
      I2 => r_addr1(1),
      I3 => r_addr1(0),
      I4 => r_addr1(2),
      I5 => reg_data10(0),
      O => reg_data1(0)
    );
\reg_data1[10]_INST_0\: unisim.vcomponents.LUT6
    generic map(
      INIT => X"FFFFFFFE00000000"
    )
        port map (
      I0 => r_addr1(4),
      I1 => r_addr1(3),
      I2 => r_addr1(1),
      I3 => r_addr1(0),
      I4 => r_addr1(2),
      I5 => reg_data10(10),
      O => reg_data1(10)
    );
\reg_data1[11]_INST_0\: unisim.vcomponents.LUT6
    generic map(
      INIT => X"FFFFFFFE00000000"
    )
        port map (
      I0 => r_addr1(4),
      I1 => r_addr1(3),
      I2 => r_addr1(1),
      I3 => r_addr1(0),
      I4 => r_addr1(2),
      I5 => reg_data10(11),
      O => reg_data1(11)
    );
\reg_data1[12]_INST_0\: unisim.vcomponents.LUT6
    generic map(
      INIT => X"FFFFFFFE00000000"
    )
        port map (
      I0 => r_addr1(4),
      I1 => r_addr1(3),
      I2 => r_addr1(1),
      I3 => r_addr1(0),
      I4 => r_addr1(2),
      I5 => reg_data10(12),
      O => reg_data1(12)
    );
\reg_data1[13]_INST_0\: unisim.vcomponents.LUT6
    generic map(
      INIT => X"FFFFFFFE00000000"
    )
        port map (
      I0 => r_addr1(4),
      I1 => r_addr1(3),
      I2 => r_addr1(1),
      I3 => r_addr1(0),
      I4 => r_addr1(2),
      I5 => reg_data10(13),
      O => reg_data1(13)
    );
\reg_data1[14]_INST_0\: unisim.vcomponents.LUT6
    generic map(
      INIT => X"FFFFFFFE00000000"
    )
        port map (
      I0 => r_addr1(4),
      I1 => r_addr1(3),
      I2 => r_addr1(1),
      I3 => r_addr1(0),
      I4 => r_addr1(2),
      I5 => reg_data10(14),
      O => reg_data1(14)
    );
\reg_data1[15]_INST_0\: unisim.vcomponents.LUT6
    generic map(
      INIT => X"FFFFFFFE00000000"
    )
        port map (
      I0 => r_addr1(4),
      I1 => r_addr1(3),
      I2 => r_addr1(1),
      I3 => r_addr1(0),
      I4 => r_addr1(2),
      I5 => reg_data10(15),
      O => reg_data1(15)
    );
\reg_data1[16]_INST_0\: unisim.vcomponents.LUT6
    generic map(
      INIT => X"FFFFFFFE00000000"
    )
        port map (
      I0 => r_addr1(4),
      I1 => r_addr1(3),
      I2 => r_addr1(1),
      I3 => r_addr1(0),
      I4 => r_addr1(2),
      I5 => reg_data10(16),
      O => reg_data1(16)
    );
\reg_data1[17]_INST_0\: unisim.vcomponents.LUT6
    generic map(
      INIT => X"FFFFFFFE00000000"
    )
        port map (
      I0 => r_addr1(4),
      I1 => r_addr1(3),
      I2 => r_addr1(1),
      I3 => r_addr1(0),
      I4 => r_addr1(2),
      I5 => reg_data10(17),
      O => reg_data1(17)
    );
\reg_data1[18]_INST_0\: unisim.vcomponents.LUT6
    generic map(
      INIT => X"FFFFFFFE00000000"
    )
        port map (
      I0 => r_addr1(4),
      I1 => r_addr1(3),
      I2 => r_addr1(1),
      I3 => r_addr1(0),
      I4 => r_addr1(2),
      I5 => reg_data10(18),
      O => reg_data1(18)
    );
\reg_data1[19]_INST_0\: unisim.vcomponents.LUT6
    generic map(
      INIT => X"FFFFFFFE00000000"
    )
        port map (
      I0 => r_addr1(4),
      I1 => r_addr1(3),
      I2 => r_addr1(1),
      I3 => r_addr1(0),
      I4 => r_addr1(2),
      I5 => reg_data10(19),
      O => reg_data1(19)
    );
\reg_data1[1]_INST_0\: unisim.vcomponents.LUT6
    generic map(
      INIT => X"FFFFFFFE00000000"
    )
        port map (
      I0 => r_addr1(4),
      I1 => r_addr1(3),
      I2 => r_addr1(1),
      I3 => r_addr1(0),
      I4 => r_addr1(2),
      I5 => reg_data10(1),
      O => reg_data1(1)
    );
\reg_data1[20]_INST_0\: unisim.vcomponents.LUT6
    generic map(
      INIT => X"FFFFFFFE00000000"
    )
        port map (
      I0 => r_addr1(4),
      I1 => r_addr1(3),
      I2 => r_addr1(1),
      I3 => r_addr1(0),
      I4 => r_addr1(2),
      I5 => reg_data10(20),
      O => reg_data1(20)
    );
\reg_data1[21]_INST_0\: unisim.vcomponents.LUT6
    generic map(
      INIT => X"FFFFFFFE00000000"
    )
        port map (
      I0 => r_addr1(4),
      I1 => r_addr1(3),
      I2 => r_addr1(1),
      I3 => r_addr1(0),
      I4 => r_addr1(2),
      I5 => reg_data10(21),
      O => reg_data1(21)
    );
\reg_data1[22]_INST_0\: unisim.vcomponents.LUT6
    generic map(
      INIT => X"FFFFFFFE00000000"
    )
        port map (
      I0 => r_addr1(4),
      I1 => r_addr1(3),
      I2 => r_addr1(1),
      I3 => r_addr1(0),
      I4 => r_addr1(2),
      I5 => reg_data10(22),
      O => reg_data1(22)
    );
\reg_data1[23]_INST_0\: unisim.vcomponents.LUT6
    generic map(
      INIT => X"FFFFFFFE00000000"
    )
        port map (
      I0 => r_addr1(4),
      I1 => r_addr1(3),
      I2 => r_addr1(1),
      I3 => r_addr1(0),
      I4 => r_addr1(2),
      I5 => reg_data10(23),
      O => reg_data1(23)
    );
\reg_data1[24]_INST_0\: unisim.vcomponents.LUT6
    generic map(
      INIT => X"FFFFFFFE00000000"
    )
        port map (
      I0 => r_addr1(4),
      I1 => r_addr1(3),
      I2 => r_addr1(1),
      I3 => r_addr1(0),
      I4 => r_addr1(2),
      I5 => reg_data10(24),
      O => reg_data1(24)
    );
\reg_data1[25]_INST_0\: unisim.vcomponents.LUT6
    generic map(
      INIT => X"FFFFFFFE00000000"
    )
        port map (
      I0 => r_addr1(4),
      I1 => r_addr1(3),
      I2 => r_addr1(1),
      I3 => r_addr1(0),
      I4 => r_addr1(2),
      I5 => reg_data10(25),
      O => reg_data1(25)
    );
\reg_data1[26]_INST_0\: unisim.vcomponents.LUT6
    generic map(
      INIT => X"FFFFFFFE00000000"
    )
        port map (
      I0 => r_addr1(4),
      I1 => r_addr1(3),
      I2 => r_addr1(1),
      I3 => r_addr1(0),
      I4 => r_addr1(2),
      I5 => reg_data10(26),
      O => reg_data1(26)
    );
\reg_data1[27]_INST_0\: unisim.vcomponents.LUT6
    generic map(
      INIT => X"FFFFFFFE00000000"
    )
        port map (
      I0 => r_addr1(4),
      I1 => r_addr1(3),
      I2 => r_addr1(1),
      I3 => r_addr1(0),
      I4 => r_addr1(2),
      I5 => reg_data10(27),
      O => reg_data1(27)
    );
\reg_data1[28]_INST_0\: unisim.vcomponents.LUT6
    generic map(
      INIT => X"FFFFFFFE00000000"
    )
        port map (
      I0 => r_addr1(4),
      I1 => r_addr1(3),
      I2 => r_addr1(1),
      I3 => r_addr1(0),
      I4 => r_addr1(2),
      I5 => reg_data10(28),
      O => reg_data1(28)
    );
\reg_data1[29]_INST_0\: unisim.vcomponents.LUT6
    generic map(
      INIT => X"FFFFFFFE00000000"
    )
        port map (
      I0 => r_addr1(4),
      I1 => r_addr1(3),
      I2 => r_addr1(1),
      I3 => r_addr1(0),
      I4 => r_addr1(2),
      I5 => reg_data10(29),
      O => reg_data1(29)
    );
\reg_data1[2]_INST_0\: unisim.vcomponents.LUT6
    generic map(
      INIT => X"FFFFFFFE00000000"
    )
        port map (
      I0 => r_addr1(4),
      I1 => r_addr1(3),
      I2 => r_addr1(1),
      I3 => r_addr1(0),
      I4 => r_addr1(2),
      I5 => reg_data10(2),
      O => reg_data1(2)
    );
\reg_data1[30]_INST_0\: unisim.vcomponents.LUT6
    generic map(
      INIT => X"FFFFFFFE00000000"
    )
        port map (
      I0 => r_addr1(4),
      I1 => r_addr1(3),
      I2 => r_addr1(1),
      I3 => r_addr1(0),
      I4 => r_addr1(2),
      I5 => reg_data10(30),
      O => reg_data1(30)
    );
\reg_data1[31]_INST_0\: unisim.vcomponents.LUT6
    generic map(
      INIT => X"FFFFFFFE00000000"
    )
        port map (
      I0 => r_addr1(4),
      I1 => r_addr1(3),
      I2 => r_addr1(1),
      I3 => r_addr1(0),
      I4 => r_addr1(2),
      I5 => reg_data10(31),
      O => reg_data1(31)
    );
\reg_data1[3]_INST_0\: unisim.vcomponents.LUT6
    generic map(
      INIT => X"FFFFFFFE00000000"
    )
        port map (
      I0 => r_addr1(4),
      I1 => r_addr1(3),
      I2 => r_addr1(1),
      I3 => r_addr1(0),
      I4 => r_addr1(2),
      I5 => reg_data10(3),
      O => reg_data1(3)
    );
\reg_data1[4]_INST_0\: unisim.vcomponents.LUT6
    generic map(
      INIT => X"FFFFFFFE00000000"
    )
        port map (
      I0 => r_addr1(4),
      I1 => r_addr1(3),
      I2 => r_addr1(1),
      I3 => r_addr1(0),
      I4 => r_addr1(2),
      I5 => reg_data10(4),
      O => reg_data1(4)
    );
\reg_data1[5]_INST_0\: unisim.vcomponents.LUT6
    generic map(
      INIT => X"FFFFFFFE00000000"
    )
        port map (
      I0 => r_addr1(4),
      I1 => r_addr1(3),
      I2 => r_addr1(1),
      I3 => r_addr1(0),
      I4 => r_addr1(2),
      I5 => reg_data10(5),
      O => reg_data1(5)
    );
\reg_data1[6]_INST_0\: unisim.vcomponents.LUT6
    generic map(
      INIT => X"FFFFFFFE00000000"
    )
        port map (
      I0 => r_addr1(4),
      I1 => r_addr1(3),
      I2 => r_addr1(1),
      I3 => r_addr1(0),
      I4 => r_addr1(2),
      I5 => reg_data10(6),
      O => reg_data1(6)
    );
\reg_data1[7]_INST_0\: unisim.vcomponents.LUT6
    generic map(
      INIT => X"FFFFFFFE00000000"
    )
        port map (
      I0 => r_addr1(4),
      I1 => r_addr1(3),
      I2 => r_addr1(1),
      I3 => r_addr1(0),
      I4 => r_addr1(2),
      I5 => reg_data10(7),
      O => reg_data1(7)
    );
\reg_data1[8]_INST_0\: unisim.vcomponents.LUT6
    generic map(
      INIT => X"FFFFFFFE00000000"
    )
        port map (
      I0 => r_addr1(4),
      I1 => r_addr1(3),
      I2 => r_addr1(1),
      I3 => r_addr1(0),
      I4 => r_addr1(2),
      I5 => reg_data10(8),
      O => reg_data1(8)
    );
\reg_data1[9]_INST_0\: unisim.vcomponents.LUT6
    generic map(
      INIT => X"FFFFFFFE00000000"
    )
        port map (
      I0 => r_addr1(4),
      I1 => r_addr1(3),
      I2 => r_addr1(1),
      I3 => r_addr1(0),
      I4 => r_addr1(2),
      I5 => reg_data10(9),
      O => reg_data1(9)
    );
\reg_data2[0]_INST_0\: unisim.vcomponents.LUT6
    generic map(
      INIT => X"FFFFFFFE00000000"
    )
        port map (
      I0 => r_addr2(4),
      I1 => r_addr2(3),
      I2 => r_addr2(1),
      I3 => r_addr2(0),
      I4 => r_addr2(2),
      I5 => reg_data20(0),
      O => reg_data2(0)
    );
\reg_data2[10]_INST_0\: unisim.vcomponents.LUT6
    generic map(
      INIT => X"FFFFFFFE00000000"
    )
        port map (
      I0 => r_addr2(4),
      I1 => r_addr2(3),
      I2 => r_addr2(1),
      I3 => r_addr2(0),
      I4 => r_addr2(2),
      I5 => reg_data20(10),
      O => reg_data2(10)
    );
\reg_data2[11]_INST_0\: unisim.vcomponents.LUT6
    generic map(
      INIT => X"FFFFFFFE00000000"
    )
        port map (
      I0 => r_addr2(4),
      I1 => r_addr2(3),
      I2 => r_addr2(1),
      I3 => r_addr2(0),
      I4 => r_addr2(2),
      I5 => reg_data20(11),
      O => reg_data2(11)
    );
\reg_data2[12]_INST_0\: unisim.vcomponents.LUT6
    generic map(
      INIT => X"FFFFFFFE00000000"
    )
        port map (
      I0 => r_addr2(4),
      I1 => r_addr2(3),
      I2 => r_addr2(1),
      I3 => r_addr2(0),
      I4 => r_addr2(2),
      I5 => reg_data20(12),
      O => reg_data2(12)
    );
\reg_data2[13]_INST_0\: unisim.vcomponents.LUT6
    generic map(
      INIT => X"FFFFFFFE00000000"
    )
        port map (
      I0 => r_addr2(4),
      I1 => r_addr2(3),
      I2 => r_addr2(1),
      I3 => r_addr2(0),
      I4 => r_addr2(2),
      I5 => reg_data20(13),
      O => reg_data2(13)
    );
\reg_data2[14]_INST_0\: unisim.vcomponents.LUT6
    generic map(
      INIT => X"FFFFFFFE00000000"
    )
        port map (
      I0 => r_addr2(4),
      I1 => r_addr2(3),
      I2 => r_addr2(1),
      I3 => r_addr2(0),
      I4 => r_addr2(2),
      I5 => reg_data20(14),
      O => reg_data2(14)
    );
\reg_data2[15]_INST_0\: unisim.vcomponents.LUT6
    generic map(
      INIT => X"FFFFFFFE00000000"
    )
        port map (
      I0 => r_addr2(4),
      I1 => r_addr2(3),
      I2 => r_addr2(1),
      I3 => r_addr2(0),
      I4 => r_addr2(2),
      I5 => reg_data20(15),
      O => reg_data2(15)
    );
\reg_data2[16]_INST_0\: unisim.vcomponents.LUT6
    generic map(
      INIT => X"FFFFFFFE00000000"
    )
        port map (
      I0 => r_addr2(4),
      I1 => r_addr2(3),
      I2 => r_addr2(1),
      I3 => r_addr2(0),
      I4 => r_addr2(2),
      I5 => reg_data20(16),
      O => reg_data2(16)
    );
\reg_data2[17]_INST_0\: unisim.vcomponents.LUT6
    generic map(
      INIT => X"FFFFFFFE00000000"
    )
        port map (
      I0 => r_addr2(4),
      I1 => r_addr2(3),
      I2 => r_addr2(1),
      I3 => r_addr2(0),
      I4 => r_addr2(2),
      I5 => reg_data20(17),
      O => reg_data2(17)
    );
\reg_data2[18]_INST_0\: unisim.vcomponents.LUT6
    generic map(
      INIT => X"FFFFFFFE00000000"
    )
        port map (
      I0 => r_addr2(4),
      I1 => r_addr2(3),
      I2 => r_addr2(1),
      I3 => r_addr2(0),
      I4 => r_addr2(2),
      I5 => reg_data20(18),
      O => reg_data2(18)
    );
\reg_data2[19]_INST_0\: unisim.vcomponents.LUT6
    generic map(
      INIT => X"FFFFFFFE00000000"
    )
        port map (
      I0 => r_addr2(4),
      I1 => r_addr2(3),
      I2 => r_addr2(1),
      I3 => r_addr2(0),
      I4 => r_addr2(2),
      I5 => reg_data20(19),
      O => reg_data2(19)
    );
\reg_data2[1]_INST_0\: unisim.vcomponents.LUT6
    generic map(
      INIT => X"FFFFFFFE00000000"
    )
        port map (
      I0 => r_addr2(4),
      I1 => r_addr2(3),
      I2 => r_addr2(1),
      I3 => r_addr2(0),
      I4 => r_addr2(2),
      I5 => reg_data20(1),
      O => reg_data2(1)
    );
\reg_data2[20]_INST_0\: unisim.vcomponents.LUT6
    generic map(
      INIT => X"FFFFFFFE00000000"
    )
        port map (
      I0 => r_addr2(4),
      I1 => r_addr2(3),
      I2 => r_addr2(1),
      I3 => r_addr2(0),
      I4 => r_addr2(2),
      I5 => reg_data20(20),
      O => reg_data2(20)
    );
\reg_data2[21]_INST_0\: unisim.vcomponents.LUT6
    generic map(
      INIT => X"FFFFFFFE00000000"
    )
        port map (
      I0 => r_addr2(4),
      I1 => r_addr2(3),
      I2 => r_addr2(1),
      I3 => r_addr2(0),
      I4 => r_addr2(2),
      I5 => reg_data20(21),
      O => reg_data2(21)
    );
\reg_data2[22]_INST_0\: unisim.vcomponents.LUT6
    generic map(
      INIT => X"FFFFFFFE00000000"
    )
        port map (
      I0 => r_addr2(4),
      I1 => r_addr2(3),
      I2 => r_addr2(1),
      I3 => r_addr2(0),
      I4 => r_addr2(2),
      I5 => reg_data20(22),
      O => reg_data2(22)
    );
\reg_data2[23]_INST_0\: unisim.vcomponents.LUT6
    generic map(
      INIT => X"FFFFFFFE00000000"
    )
        port map (
      I0 => r_addr2(4),
      I1 => r_addr2(3),
      I2 => r_addr2(1),
      I3 => r_addr2(0),
      I4 => r_addr2(2),
      I5 => reg_data20(23),
      O => reg_data2(23)
    );
\reg_data2[24]_INST_0\: unisim.vcomponents.LUT6
    generic map(
      INIT => X"FFFFFFFE00000000"
    )
        port map (
      I0 => r_addr2(4),
      I1 => r_addr2(3),
      I2 => r_addr2(1),
      I3 => r_addr2(0),
      I4 => r_addr2(2),
      I5 => reg_data20(24),
      O => reg_data2(24)
    );
\reg_data2[25]_INST_0\: unisim.vcomponents.LUT6
    generic map(
      INIT => X"FFFFFFFE00000000"
    )
        port map (
      I0 => r_addr2(4),
      I1 => r_addr2(3),
      I2 => r_addr2(1),
      I3 => r_addr2(0),
      I4 => r_addr2(2),
      I5 => reg_data20(25),
      O => reg_data2(25)
    );
\reg_data2[26]_INST_0\: unisim.vcomponents.LUT6
    generic map(
      INIT => X"FFFFFFFE00000000"
    )
        port map (
      I0 => r_addr2(4),
      I1 => r_addr2(3),
      I2 => r_addr2(1),
      I3 => r_addr2(0),
      I4 => r_addr2(2),
      I5 => reg_data20(26),
      O => reg_data2(26)
    );
\reg_data2[27]_INST_0\: unisim.vcomponents.LUT6
    generic map(
      INIT => X"FFFFFFFE00000000"
    )
        port map (
      I0 => r_addr2(4),
      I1 => r_addr2(3),
      I2 => r_addr2(1),
      I3 => r_addr2(0),
      I4 => r_addr2(2),
      I5 => reg_data20(27),
      O => reg_data2(27)
    );
\reg_data2[28]_INST_0\: unisim.vcomponents.LUT6
    generic map(
      INIT => X"FFFFFFFE00000000"
    )
        port map (
      I0 => r_addr2(4),
      I1 => r_addr2(3),
      I2 => r_addr2(1),
      I3 => r_addr2(0),
      I4 => r_addr2(2),
      I5 => reg_data20(28),
      O => reg_data2(28)
    );
\reg_data2[29]_INST_0\: unisim.vcomponents.LUT6
    generic map(
      INIT => X"FFFFFFFE00000000"
    )
        port map (
      I0 => r_addr2(4),
      I1 => r_addr2(3),
      I2 => r_addr2(1),
      I3 => r_addr2(0),
      I4 => r_addr2(2),
      I5 => reg_data20(29),
      O => reg_data2(29)
    );
\reg_data2[2]_INST_0\: unisim.vcomponents.LUT6
    generic map(
      INIT => X"FFFFFFFE00000000"
    )
        port map (
      I0 => r_addr2(4),
      I1 => r_addr2(3),
      I2 => r_addr2(1),
      I3 => r_addr2(0),
      I4 => r_addr2(2),
      I5 => reg_data20(2),
      O => reg_data2(2)
    );
\reg_data2[30]_INST_0\: unisim.vcomponents.LUT6
    generic map(
      INIT => X"FFFFFFFE00000000"
    )
        port map (
      I0 => r_addr2(4),
      I1 => r_addr2(3),
      I2 => r_addr2(1),
      I3 => r_addr2(0),
      I4 => r_addr2(2),
      I5 => reg_data20(30),
      O => reg_data2(30)
    );
\reg_data2[31]_INST_0\: unisim.vcomponents.LUT6
    generic map(
      INIT => X"FFFFFFFE00000000"
    )
        port map (
      I0 => r_addr2(4),
      I1 => r_addr2(3),
      I2 => r_addr2(1),
      I3 => r_addr2(0),
      I4 => r_addr2(2),
      I5 => reg_data20(31),
      O => reg_data2(31)
    );
\reg_data2[3]_INST_0\: unisim.vcomponents.LUT6
    generic map(
      INIT => X"FFFFFFFE00000000"
    )
        port map (
      I0 => r_addr2(4),
      I1 => r_addr2(3),
      I2 => r_addr2(1),
      I3 => r_addr2(0),
      I4 => r_addr2(2),
      I5 => reg_data20(3),
      O => reg_data2(3)
    );
\reg_data2[4]_INST_0\: unisim.vcomponents.LUT6
    generic map(
      INIT => X"FFFFFFFE00000000"
    )
        port map (
      I0 => r_addr2(4),
      I1 => r_addr2(3),
      I2 => r_addr2(1),
      I3 => r_addr2(0),
      I4 => r_addr2(2),
      I5 => reg_data20(4),
      O => reg_data2(4)
    );
\reg_data2[5]_INST_0\: unisim.vcomponents.LUT6
    generic map(
      INIT => X"FFFFFFFE00000000"
    )
        port map (
      I0 => r_addr2(4),
      I1 => r_addr2(3),
      I2 => r_addr2(1),
      I3 => r_addr2(0),
      I4 => r_addr2(2),
      I5 => reg_data20(5),
      O => reg_data2(5)
    );
\reg_data2[6]_INST_0\: unisim.vcomponents.LUT6
    generic map(
      INIT => X"FFFFFFFE00000000"
    )
        port map (
      I0 => r_addr2(4),
      I1 => r_addr2(3),
      I2 => r_addr2(1),
      I3 => r_addr2(0),
      I4 => r_addr2(2),
      I5 => reg_data20(6),
      O => reg_data2(6)
    );
\reg_data2[7]_INST_0\: unisim.vcomponents.LUT6
    generic map(
      INIT => X"FFFFFFFE00000000"
    )
        port map (
      I0 => r_addr2(4),
      I1 => r_addr2(3),
      I2 => r_addr2(1),
      I3 => r_addr2(0),
      I4 => r_addr2(2),
      I5 => reg_data20(7),
      O => reg_data2(7)
    );
\reg_data2[8]_INST_0\: unisim.vcomponents.LUT6
    generic map(
      INIT => X"FFFFFFFE00000000"
    )
        port map (
      I0 => r_addr2(4),
      I1 => r_addr2(3),
      I2 => r_addr2(1),
      I3 => r_addr2(0),
      I4 => r_addr2(2),
      I5 => reg_data20(8),
      O => reg_data2(8)
    );
\reg_data2[9]_INST_0\: unisim.vcomponents.LUT6
    generic map(
      INIT => X"FFFFFFFE00000000"
    )
        port map (
      I0 => r_addr2(4),
      I1 => r_addr2(3),
      I2 => r_addr2(1),
      I3 => r_addr2(0),
      I4 => r_addr2(2),
      I5 => reg_data20(9),
      O => reg_data2(9)
    );
end STRUCTURE;
library IEEE;
use IEEE.STD_LOGIC_1164.ALL;
library UNISIM;
use UNISIM.VCOMPONENTS.ALL;
entity decalper_eb_ot_sdeen_pot_pi_dehcac_xnilix is
  port (
    clk : in STD_LOGIC;
    w_ena : in STD_LOGIC;
    r_addr1 : in STD_LOGIC_VECTOR ( 4 downto 0 );
    r_addr2 : in STD_LOGIC_VECTOR ( 4 downto 0 );
    w_addr : in STD_LOGIC_VECTOR ( 4 downto 0 );
    w_data : in STD_LOGIC_VECTOR ( 31 downto 0 );
    reg_data1 : out STD_LOGIC_VECTOR ( 31 downto 0 );
    reg_data2 : out STD_LOGIC_VECTOR ( 31 downto 0 )
  );
  attribute NotValidForBitStream : boolean;
  attribute NotValidForBitStream of decalper_eb_ot_sdeen_pot_pi_dehcac_xnilix : entity is true;
  attribute CHECK_LICENSE_TYPE : string;
  attribute CHECK_LICENSE_TYPE of decalper_eb_ot_sdeen_pot_pi_dehcac_xnilix : entity is "risc32_regfile_0_0,regfile,{}";
  attribute DowngradeIPIdentifiedWarnings : string;
  attribute DowngradeIPIdentifiedWarnings of decalper_eb_ot_sdeen_pot_pi_dehcac_xnilix : entity is "yes";
  attribute IP_DEFINITION_SOURCE : string;
  attribute IP_DEFINITION_SOURCE of decalper_eb_ot_sdeen_pot_pi_dehcac_xnilix : entity is "module_ref";
  attribute X_CORE_INFO : string;
  attribute X_CORE_INFO of decalper_eb_ot_sdeen_pot_pi_dehcac_xnilix : entity is "regfile,Vivado 2022.2";
end decalper_eb_ot_sdeen_pot_pi_dehcac_xnilix;

architecture STRUCTURE of decalper_eb_ot_sdeen_pot_pi_dehcac_xnilix is
  attribute X_INTERFACE_INFO : string;
  attribute X_INTERFACE_INFO of clk : signal is "xilinx.com:signal:clock:1.0 clk CLK";
  attribute X_INTERFACE_PARAMETER : string;
  attribute X_INTERFACE_PARAMETER of clk : signal is "XIL_INTERFACENAME clk, FREQ_HZ 100000000, FREQ_TOLERANCE_HZ 0, PHASE 0.0, CLK_DOMAIN risc32_clk_0, INSERT_VIP 0";
begin
inst: entity work.decalper_eb_ot_sdeen_pot_pi_dehcac_xnilix_regfile
     port map (
      clk => clk,
      r_addr1(4 downto 0) => r_addr1(4 downto 0),
      r_addr2(4 downto 0) => r_addr2(4 downto 0),
      reg_data1(31 downto 0) => reg_data1(31 downto 0),
      reg_data2(31 downto 0) => reg_data2(31 downto 0),
      w_addr(4 downto 0) => w_addr(4 downto 0),
      w_data(31 downto 0) => w_data(31 downto 0),
      w_ena => w_ena
    );
end STRUCTURE;
