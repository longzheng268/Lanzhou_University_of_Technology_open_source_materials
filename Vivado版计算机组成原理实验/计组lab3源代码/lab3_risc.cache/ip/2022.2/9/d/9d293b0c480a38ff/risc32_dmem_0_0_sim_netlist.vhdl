-- Copyright 1986-2022 Xilinx, Inc. All Rights Reserved.
-- --------------------------------------------------------------------------------
-- Tool Version: Vivado v.2022.2 (win64) Build 3671981 Fri Oct 14 05:00:03 MDT 2022
-- Date        : Mon Oct 28 16:04:11 2024
-- Host        : cop running 64-bit major release  (build 9200)
-- Command     : write_vhdl -force -mode funcsim -rename_top decalper_eb_ot_sdeen_pot_pi_dehcac_xnilix -prefix
--               decalper_eb_ot_sdeen_pot_pi_dehcac_xnilix_ risc32_dmem_0_0_sim_netlist.vhdl
-- Design      : risc32_dmem_0_0
-- Purpose     : This VHDL netlist is a functional simulation representation of the design and should not be modified or
--               synthesized. This netlist cannot be used for SDF annotated simulation.
-- Device      : xc7a100tcsg324-1
-- --------------------------------------------------------------------------------
library IEEE;
use IEEE.STD_LOGIC_1164.ALL;
library UNISIM;
use UNISIM.VCOMPONENTS.ALL;
entity decalper_eb_ot_sdeen_pot_pi_dehcac_xnilix_dmem is
  port (
    d_out : out STD_LOGIC_VECTOR ( 31 downto 0 );
    w_ena : in STD_LOGIC;
    addr : in STD_LOGIC_VECTOR ( 7 downto 0 );
    clk : in STD_LOGIC;
    d_in : in STD_LOGIC_VECTOR ( 31 downto 0 )
  );
end decalper_eb_ot_sdeen_pot_pi_dehcac_xnilix_dmem;

architecture STRUCTURE of decalper_eb_ot_sdeen_pot_pi_dehcac_xnilix_dmem is
  signal \mem_reg_0_127_0_0__0_n_0\ : STD_LOGIC;
  signal \mem_reg_0_127_0_0__10_n_0\ : STD_LOGIC;
  signal \mem_reg_0_127_0_0__11_n_0\ : STD_LOGIC;
  signal \mem_reg_0_127_0_0__12_n_0\ : STD_LOGIC;
  signal \mem_reg_0_127_0_0__13_n_0\ : STD_LOGIC;
  signal \mem_reg_0_127_0_0__14_n_0\ : STD_LOGIC;
  signal \mem_reg_0_127_0_0__15_n_0\ : STD_LOGIC;
  signal \mem_reg_0_127_0_0__16_n_0\ : STD_LOGIC;
  signal \mem_reg_0_127_0_0__17_n_0\ : STD_LOGIC;
  signal \mem_reg_0_127_0_0__18_n_0\ : STD_LOGIC;
  signal \mem_reg_0_127_0_0__19_n_0\ : STD_LOGIC;
  signal \mem_reg_0_127_0_0__1_n_0\ : STD_LOGIC;
  signal \mem_reg_0_127_0_0__20_n_0\ : STD_LOGIC;
  signal \mem_reg_0_127_0_0__21_n_0\ : STD_LOGIC;
  signal \mem_reg_0_127_0_0__22_n_0\ : STD_LOGIC;
  signal \mem_reg_0_127_0_0__23_n_0\ : STD_LOGIC;
  signal \mem_reg_0_127_0_0__24_n_0\ : STD_LOGIC;
  signal \mem_reg_0_127_0_0__25_n_0\ : STD_LOGIC;
  signal \mem_reg_0_127_0_0__26_n_0\ : STD_LOGIC;
  signal \mem_reg_0_127_0_0__27_n_0\ : STD_LOGIC;
  signal \mem_reg_0_127_0_0__28_n_0\ : STD_LOGIC;
  signal \mem_reg_0_127_0_0__29_n_0\ : STD_LOGIC;
  signal \mem_reg_0_127_0_0__2_n_0\ : STD_LOGIC;
  signal \mem_reg_0_127_0_0__30_n_0\ : STD_LOGIC;
  signal \mem_reg_0_127_0_0__3_n_0\ : STD_LOGIC;
  signal \mem_reg_0_127_0_0__4_n_0\ : STD_LOGIC;
  signal \mem_reg_0_127_0_0__5_n_0\ : STD_LOGIC;
  signal \mem_reg_0_127_0_0__6_n_0\ : STD_LOGIC;
  signal \mem_reg_0_127_0_0__7_n_0\ : STD_LOGIC;
  signal \mem_reg_0_127_0_0__8_n_0\ : STD_LOGIC;
  signal \mem_reg_0_127_0_0__9_n_0\ : STD_LOGIC;
  signal mem_reg_0_127_0_0_i_1_n_0 : STD_LOGIC;
  signal mem_reg_0_127_0_0_n_0 : STD_LOGIC;
  signal \mem_reg_0_15_0_0__0_n_0\ : STD_LOGIC;
  signal \mem_reg_0_15_0_0__10_n_0\ : STD_LOGIC;
  signal \mem_reg_0_15_0_0__11_n_0\ : STD_LOGIC;
  signal \mem_reg_0_15_0_0__12_n_0\ : STD_LOGIC;
  signal \mem_reg_0_15_0_0__13_n_0\ : STD_LOGIC;
  signal \mem_reg_0_15_0_0__14_n_0\ : STD_LOGIC;
  signal \mem_reg_0_15_0_0__15_n_0\ : STD_LOGIC;
  signal \mem_reg_0_15_0_0__16_n_0\ : STD_LOGIC;
  signal \mem_reg_0_15_0_0__17_n_0\ : STD_LOGIC;
  signal \mem_reg_0_15_0_0__18_n_0\ : STD_LOGIC;
  signal \mem_reg_0_15_0_0__19_n_0\ : STD_LOGIC;
  signal \mem_reg_0_15_0_0__1_n_0\ : STD_LOGIC;
  signal \mem_reg_0_15_0_0__20_n_0\ : STD_LOGIC;
  signal \mem_reg_0_15_0_0__21_n_0\ : STD_LOGIC;
  signal \mem_reg_0_15_0_0__22_n_0\ : STD_LOGIC;
  signal \mem_reg_0_15_0_0__23_n_0\ : STD_LOGIC;
  signal \mem_reg_0_15_0_0__24_n_0\ : STD_LOGIC;
  signal \mem_reg_0_15_0_0__25_n_0\ : STD_LOGIC;
  signal \mem_reg_0_15_0_0__26_n_0\ : STD_LOGIC;
  signal \mem_reg_0_15_0_0__27_n_0\ : STD_LOGIC;
  signal \mem_reg_0_15_0_0__28_n_0\ : STD_LOGIC;
  signal \mem_reg_0_15_0_0__29_n_0\ : STD_LOGIC;
  signal \mem_reg_0_15_0_0__2_n_0\ : STD_LOGIC;
  signal \mem_reg_0_15_0_0__30_n_0\ : STD_LOGIC;
  signal \mem_reg_0_15_0_0__3_n_0\ : STD_LOGIC;
  signal \mem_reg_0_15_0_0__4_n_0\ : STD_LOGIC;
  signal \mem_reg_0_15_0_0__5_n_0\ : STD_LOGIC;
  signal \mem_reg_0_15_0_0__6_n_0\ : STD_LOGIC;
  signal \mem_reg_0_15_0_0__7_n_0\ : STD_LOGIC;
  signal \mem_reg_0_15_0_0__8_n_0\ : STD_LOGIC;
  signal \mem_reg_0_15_0_0__9_n_0\ : STD_LOGIC;
  signal mem_reg_0_15_0_0_i_1_n_0 : STD_LOGIC;
  signal mem_reg_0_15_0_0_n_0 : STD_LOGIC;
  attribute RTL_RAM_BITS : integer;
  attribute RTL_RAM_BITS of mem_reg_0_127_0_0 : label is 4128;
  attribute RTL_RAM_NAME : string;
  attribute RTL_RAM_NAME of mem_reg_0_127_0_0 : label is "inst/mem_reg";
  attribute RTL_RAM_TYPE : string;
  attribute RTL_RAM_TYPE of mem_reg_0_127_0_0 : label is "RAM_SP";
  attribute ram_addr_begin : integer;
  attribute ram_addr_begin of mem_reg_0_127_0_0 : label is 0;
  attribute ram_addr_end : integer;
  attribute ram_addr_end of mem_reg_0_127_0_0 : label is 127;
  attribute ram_offset : integer;
  attribute ram_offset of mem_reg_0_127_0_0 : label is 0;
  attribute ram_slice_begin : integer;
  attribute ram_slice_begin of mem_reg_0_127_0_0 : label is 0;
  attribute ram_slice_end : integer;
  attribute ram_slice_end of mem_reg_0_127_0_0 : label is 0;
  attribute RTL_RAM_BITS of \mem_reg_0_127_0_0__0\ : label is 4128;
  attribute RTL_RAM_NAME of \mem_reg_0_127_0_0__0\ : label is "inst/mem_reg";
  attribute RTL_RAM_TYPE of \mem_reg_0_127_0_0__0\ : label is "RAM_SP";
  attribute ram_addr_begin of \mem_reg_0_127_0_0__0\ : label is 0;
  attribute ram_addr_end of \mem_reg_0_127_0_0__0\ : label is 127;
  attribute ram_offset of \mem_reg_0_127_0_0__0\ : label is 0;
  attribute ram_slice_begin of \mem_reg_0_127_0_0__0\ : label is 1;
  attribute ram_slice_end of \mem_reg_0_127_0_0__0\ : label is 1;
  attribute RTL_RAM_BITS of \mem_reg_0_127_0_0__1\ : label is 4128;
  attribute RTL_RAM_NAME of \mem_reg_0_127_0_0__1\ : label is "inst/mem_reg";
  attribute RTL_RAM_TYPE of \mem_reg_0_127_0_0__1\ : label is "RAM_SP";
  attribute ram_addr_begin of \mem_reg_0_127_0_0__1\ : label is 0;
  attribute ram_addr_end of \mem_reg_0_127_0_0__1\ : label is 127;
  attribute ram_offset of \mem_reg_0_127_0_0__1\ : label is 0;
  attribute ram_slice_begin of \mem_reg_0_127_0_0__1\ : label is 2;
  attribute ram_slice_end of \mem_reg_0_127_0_0__1\ : label is 2;
  attribute RTL_RAM_BITS of \mem_reg_0_127_0_0__10\ : label is 4128;
  attribute RTL_RAM_NAME of \mem_reg_0_127_0_0__10\ : label is "inst/mem_reg";
  attribute RTL_RAM_TYPE of \mem_reg_0_127_0_0__10\ : label is "RAM_SP";
  attribute ram_addr_begin of \mem_reg_0_127_0_0__10\ : label is 0;
  attribute ram_addr_end of \mem_reg_0_127_0_0__10\ : label is 127;
  attribute ram_offset of \mem_reg_0_127_0_0__10\ : label is 0;
  attribute ram_slice_begin of \mem_reg_0_127_0_0__10\ : label is 11;
  attribute ram_slice_end of \mem_reg_0_127_0_0__10\ : label is 11;
  attribute RTL_RAM_BITS of \mem_reg_0_127_0_0__11\ : label is 4128;
  attribute RTL_RAM_NAME of \mem_reg_0_127_0_0__11\ : label is "inst/mem_reg";
  attribute RTL_RAM_TYPE of \mem_reg_0_127_0_0__11\ : label is "RAM_SP";
  attribute ram_addr_begin of \mem_reg_0_127_0_0__11\ : label is 0;
  attribute ram_addr_end of \mem_reg_0_127_0_0__11\ : label is 127;
  attribute ram_offset of \mem_reg_0_127_0_0__11\ : label is 0;
  attribute ram_slice_begin of \mem_reg_0_127_0_0__11\ : label is 12;
  attribute ram_slice_end of \mem_reg_0_127_0_0__11\ : label is 12;
  attribute RTL_RAM_BITS of \mem_reg_0_127_0_0__12\ : label is 4128;
  attribute RTL_RAM_NAME of \mem_reg_0_127_0_0__12\ : label is "inst/mem_reg";
  attribute RTL_RAM_TYPE of \mem_reg_0_127_0_0__12\ : label is "RAM_SP";
  attribute ram_addr_begin of \mem_reg_0_127_0_0__12\ : label is 0;
  attribute ram_addr_end of \mem_reg_0_127_0_0__12\ : label is 127;
  attribute ram_offset of \mem_reg_0_127_0_0__12\ : label is 0;
  attribute ram_slice_begin of \mem_reg_0_127_0_0__12\ : label is 13;
  attribute ram_slice_end of \mem_reg_0_127_0_0__12\ : label is 13;
  attribute RTL_RAM_BITS of \mem_reg_0_127_0_0__13\ : label is 4128;
  attribute RTL_RAM_NAME of \mem_reg_0_127_0_0__13\ : label is "inst/mem_reg";
  attribute RTL_RAM_TYPE of \mem_reg_0_127_0_0__13\ : label is "RAM_SP";
  attribute ram_addr_begin of \mem_reg_0_127_0_0__13\ : label is 0;
  attribute ram_addr_end of \mem_reg_0_127_0_0__13\ : label is 127;
  attribute ram_offset of \mem_reg_0_127_0_0__13\ : label is 0;
  attribute ram_slice_begin of \mem_reg_0_127_0_0__13\ : label is 14;
  attribute ram_slice_end of \mem_reg_0_127_0_0__13\ : label is 14;
  attribute RTL_RAM_BITS of \mem_reg_0_127_0_0__14\ : label is 4128;
  attribute RTL_RAM_NAME of \mem_reg_0_127_0_0__14\ : label is "inst/mem_reg";
  attribute RTL_RAM_TYPE of \mem_reg_0_127_0_0__14\ : label is "RAM_SP";
  attribute ram_addr_begin of \mem_reg_0_127_0_0__14\ : label is 0;
  attribute ram_addr_end of \mem_reg_0_127_0_0__14\ : label is 127;
  attribute ram_offset of \mem_reg_0_127_0_0__14\ : label is 0;
  attribute ram_slice_begin of \mem_reg_0_127_0_0__14\ : label is 15;
  attribute ram_slice_end of \mem_reg_0_127_0_0__14\ : label is 15;
  attribute RTL_RAM_BITS of \mem_reg_0_127_0_0__15\ : label is 4128;
  attribute RTL_RAM_NAME of \mem_reg_0_127_0_0__15\ : label is "inst/mem_reg";
  attribute RTL_RAM_TYPE of \mem_reg_0_127_0_0__15\ : label is "RAM_SP";
  attribute ram_addr_begin of \mem_reg_0_127_0_0__15\ : label is 0;
  attribute ram_addr_end of \mem_reg_0_127_0_0__15\ : label is 127;
  attribute ram_offset of \mem_reg_0_127_0_0__15\ : label is 0;
  attribute ram_slice_begin of \mem_reg_0_127_0_0__15\ : label is 16;
  attribute ram_slice_end of \mem_reg_0_127_0_0__15\ : label is 16;
  attribute RTL_RAM_BITS of \mem_reg_0_127_0_0__16\ : label is 4128;
  attribute RTL_RAM_NAME of \mem_reg_0_127_0_0__16\ : label is "inst/mem_reg";
  attribute RTL_RAM_TYPE of \mem_reg_0_127_0_0__16\ : label is "RAM_SP";
  attribute ram_addr_begin of \mem_reg_0_127_0_0__16\ : label is 0;
  attribute ram_addr_end of \mem_reg_0_127_0_0__16\ : label is 127;
  attribute ram_offset of \mem_reg_0_127_0_0__16\ : label is 0;
  attribute ram_slice_begin of \mem_reg_0_127_0_0__16\ : label is 17;
  attribute ram_slice_end of \mem_reg_0_127_0_0__16\ : label is 17;
  attribute RTL_RAM_BITS of \mem_reg_0_127_0_0__17\ : label is 4128;
  attribute RTL_RAM_NAME of \mem_reg_0_127_0_0__17\ : label is "inst/mem_reg";
  attribute RTL_RAM_TYPE of \mem_reg_0_127_0_0__17\ : label is "RAM_SP";
  attribute ram_addr_begin of \mem_reg_0_127_0_0__17\ : label is 0;
  attribute ram_addr_end of \mem_reg_0_127_0_0__17\ : label is 127;
  attribute ram_offset of \mem_reg_0_127_0_0__17\ : label is 0;
  attribute ram_slice_begin of \mem_reg_0_127_0_0__17\ : label is 18;
  attribute ram_slice_end of \mem_reg_0_127_0_0__17\ : label is 18;
  attribute RTL_RAM_BITS of \mem_reg_0_127_0_0__18\ : label is 4128;
  attribute RTL_RAM_NAME of \mem_reg_0_127_0_0__18\ : label is "inst/mem_reg";
  attribute RTL_RAM_TYPE of \mem_reg_0_127_0_0__18\ : label is "RAM_SP";
  attribute ram_addr_begin of \mem_reg_0_127_0_0__18\ : label is 0;
  attribute ram_addr_end of \mem_reg_0_127_0_0__18\ : label is 127;
  attribute ram_offset of \mem_reg_0_127_0_0__18\ : label is 0;
  attribute ram_slice_begin of \mem_reg_0_127_0_0__18\ : label is 19;
  attribute ram_slice_end of \mem_reg_0_127_0_0__18\ : label is 19;
  attribute RTL_RAM_BITS of \mem_reg_0_127_0_0__19\ : label is 4128;
  attribute RTL_RAM_NAME of \mem_reg_0_127_0_0__19\ : label is "inst/mem_reg";
  attribute RTL_RAM_TYPE of \mem_reg_0_127_0_0__19\ : label is "RAM_SP";
  attribute ram_addr_begin of \mem_reg_0_127_0_0__19\ : label is 0;
  attribute ram_addr_end of \mem_reg_0_127_0_0__19\ : label is 127;
  attribute ram_offset of \mem_reg_0_127_0_0__19\ : label is 0;
  attribute ram_slice_begin of \mem_reg_0_127_0_0__19\ : label is 20;
  attribute ram_slice_end of \mem_reg_0_127_0_0__19\ : label is 20;
  attribute RTL_RAM_BITS of \mem_reg_0_127_0_0__2\ : label is 4128;
  attribute RTL_RAM_NAME of \mem_reg_0_127_0_0__2\ : label is "inst/mem_reg";
  attribute RTL_RAM_TYPE of \mem_reg_0_127_0_0__2\ : label is "RAM_SP";
  attribute ram_addr_begin of \mem_reg_0_127_0_0__2\ : label is 0;
  attribute ram_addr_end of \mem_reg_0_127_0_0__2\ : label is 127;
  attribute ram_offset of \mem_reg_0_127_0_0__2\ : label is 0;
  attribute ram_slice_begin of \mem_reg_0_127_0_0__2\ : label is 3;
  attribute ram_slice_end of \mem_reg_0_127_0_0__2\ : label is 3;
  attribute RTL_RAM_BITS of \mem_reg_0_127_0_0__20\ : label is 4128;
  attribute RTL_RAM_NAME of \mem_reg_0_127_0_0__20\ : label is "inst/mem_reg";
  attribute RTL_RAM_TYPE of \mem_reg_0_127_0_0__20\ : label is "RAM_SP";
  attribute ram_addr_begin of \mem_reg_0_127_0_0__20\ : label is 0;
  attribute ram_addr_end of \mem_reg_0_127_0_0__20\ : label is 127;
  attribute ram_offset of \mem_reg_0_127_0_0__20\ : label is 0;
  attribute ram_slice_begin of \mem_reg_0_127_0_0__20\ : label is 21;
  attribute ram_slice_end of \mem_reg_0_127_0_0__20\ : label is 21;
  attribute RTL_RAM_BITS of \mem_reg_0_127_0_0__21\ : label is 4128;
  attribute RTL_RAM_NAME of \mem_reg_0_127_0_0__21\ : label is "inst/mem_reg";
  attribute RTL_RAM_TYPE of \mem_reg_0_127_0_0__21\ : label is "RAM_SP";
  attribute ram_addr_begin of \mem_reg_0_127_0_0__21\ : label is 0;
  attribute ram_addr_end of \mem_reg_0_127_0_0__21\ : label is 127;
  attribute ram_offset of \mem_reg_0_127_0_0__21\ : label is 0;
  attribute ram_slice_begin of \mem_reg_0_127_0_0__21\ : label is 22;
  attribute ram_slice_end of \mem_reg_0_127_0_0__21\ : label is 22;
  attribute RTL_RAM_BITS of \mem_reg_0_127_0_0__22\ : label is 4128;
  attribute RTL_RAM_NAME of \mem_reg_0_127_0_0__22\ : label is "inst/mem_reg";
  attribute RTL_RAM_TYPE of \mem_reg_0_127_0_0__22\ : label is "RAM_SP";
  attribute ram_addr_begin of \mem_reg_0_127_0_0__22\ : label is 0;
  attribute ram_addr_end of \mem_reg_0_127_0_0__22\ : label is 127;
  attribute ram_offset of \mem_reg_0_127_0_0__22\ : label is 0;
  attribute ram_slice_begin of \mem_reg_0_127_0_0__22\ : label is 23;
  attribute ram_slice_end of \mem_reg_0_127_0_0__22\ : label is 23;
  attribute RTL_RAM_BITS of \mem_reg_0_127_0_0__23\ : label is 4128;
  attribute RTL_RAM_NAME of \mem_reg_0_127_0_0__23\ : label is "inst/mem_reg";
  attribute RTL_RAM_TYPE of \mem_reg_0_127_0_0__23\ : label is "RAM_SP";
  attribute ram_addr_begin of \mem_reg_0_127_0_0__23\ : label is 0;
  attribute ram_addr_end of \mem_reg_0_127_0_0__23\ : label is 127;
  attribute ram_offset of \mem_reg_0_127_0_0__23\ : label is 0;
  attribute ram_slice_begin of \mem_reg_0_127_0_0__23\ : label is 24;
  attribute ram_slice_end of \mem_reg_0_127_0_0__23\ : label is 24;
  attribute RTL_RAM_BITS of \mem_reg_0_127_0_0__24\ : label is 4128;
  attribute RTL_RAM_NAME of \mem_reg_0_127_0_0__24\ : label is "inst/mem_reg";
  attribute RTL_RAM_TYPE of \mem_reg_0_127_0_0__24\ : label is "RAM_SP";
  attribute ram_addr_begin of \mem_reg_0_127_0_0__24\ : label is 0;
  attribute ram_addr_end of \mem_reg_0_127_0_0__24\ : label is 127;
  attribute ram_offset of \mem_reg_0_127_0_0__24\ : label is 0;
  attribute ram_slice_begin of \mem_reg_0_127_0_0__24\ : label is 25;
  attribute ram_slice_end of \mem_reg_0_127_0_0__24\ : label is 25;
  attribute RTL_RAM_BITS of \mem_reg_0_127_0_0__25\ : label is 4128;
  attribute RTL_RAM_NAME of \mem_reg_0_127_0_0__25\ : label is "inst/mem_reg";
  attribute RTL_RAM_TYPE of \mem_reg_0_127_0_0__25\ : label is "RAM_SP";
  attribute ram_addr_begin of \mem_reg_0_127_0_0__25\ : label is 0;
  attribute ram_addr_end of \mem_reg_0_127_0_0__25\ : label is 127;
  attribute ram_offset of \mem_reg_0_127_0_0__25\ : label is 0;
  attribute ram_slice_begin of \mem_reg_0_127_0_0__25\ : label is 26;
  attribute ram_slice_end of \mem_reg_0_127_0_0__25\ : label is 26;
  attribute RTL_RAM_BITS of \mem_reg_0_127_0_0__26\ : label is 4128;
  attribute RTL_RAM_NAME of \mem_reg_0_127_0_0__26\ : label is "inst/mem_reg";
  attribute RTL_RAM_TYPE of \mem_reg_0_127_0_0__26\ : label is "RAM_SP";
  attribute ram_addr_begin of \mem_reg_0_127_0_0__26\ : label is 0;
  attribute ram_addr_end of \mem_reg_0_127_0_0__26\ : label is 127;
  attribute ram_offset of \mem_reg_0_127_0_0__26\ : label is 0;
  attribute ram_slice_begin of \mem_reg_0_127_0_0__26\ : label is 27;
  attribute ram_slice_end of \mem_reg_0_127_0_0__26\ : label is 27;
  attribute RTL_RAM_BITS of \mem_reg_0_127_0_0__27\ : label is 4128;
  attribute RTL_RAM_NAME of \mem_reg_0_127_0_0__27\ : label is "inst/mem_reg";
  attribute RTL_RAM_TYPE of \mem_reg_0_127_0_0__27\ : label is "RAM_SP";
  attribute ram_addr_begin of \mem_reg_0_127_0_0__27\ : label is 0;
  attribute ram_addr_end of \mem_reg_0_127_0_0__27\ : label is 127;
  attribute ram_offset of \mem_reg_0_127_0_0__27\ : label is 0;
  attribute ram_slice_begin of \mem_reg_0_127_0_0__27\ : label is 28;
  attribute ram_slice_end of \mem_reg_0_127_0_0__27\ : label is 28;
  attribute RTL_RAM_BITS of \mem_reg_0_127_0_0__28\ : label is 4128;
  attribute RTL_RAM_NAME of \mem_reg_0_127_0_0__28\ : label is "inst/mem_reg";
  attribute RTL_RAM_TYPE of \mem_reg_0_127_0_0__28\ : label is "RAM_SP";
  attribute ram_addr_begin of \mem_reg_0_127_0_0__28\ : label is 0;
  attribute ram_addr_end of \mem_reg_0_127_0_0__28\ : label is 127;
  attribute ram_offset of \mem_reg_0_127_0_0__28\ : label is 0;
  attribute ram_slice_begin of \mem_reg_0_127_0_0__28\ : label is 29;
  attribute ram_slice_end of \mem_reg_0_127_0_0__28\ : label is 29;
  attribute RTL_RAM_BITS of \mem_reg_0_127_0_0__29\ : label is 4128;
  attribute RTL_RAM_NAME of \mem_reg_0_127_0_0__29\ : label is "inst/mem_reg";
  attribute RTL_RAM_TYPE of \mem_reg_0_127_0_0__29\ : label is "RAM_SP";
  attribute ram_addr_begin of \mem_reg_0_127_0_0__29\ : label is 0;
  attribute ram_addr_end of \mem_reg_0_127_0_0__29\ : label is 127;
  attribute ram_offset of \mem_reg_0_127_0_0__29\ : label is 0;
  attribute ram_slice_begin of \mem_reg_0_127_0_0__29\ : label is 30;
  attribute ram_slice_end of \mem_reg_0_127_0_0__29\ : label is 30;
  attribute RTL_RAM_BITS of \mem_reg_0_127_0_0__3\ : label is 4128;
  attribute RTL_RAM_NAME of \mem_reg_0_127_0_0__3\ : label is "inst/mem_reg";
  attribute RTL_RAM_TYPE of \mem_reg_0_127_0_0__3\ : label is "RAM_SP";
  attribute ram_addr_begin of \mem_reg_0_127_0_0__3\ : label is 0;
  attribute ram_addr_end of \mem_reg_0_127_0_0__3\ : label is 127;
  attribute ram_offset of \mem_reg_0_127_0_0__3\ : label is 0;
  attribute ram_slice_begin of \mem_reg_0_127_0_0__3\ : label is 4;
  attribute ram_slice_end of \mem_reg_0_127_0_0__3\ : label is 4;
  attribute RTL_RAM_BITS of \mem_reg_0_127_0_0__30\ : label is 4128;
  attribute RTL_RAM_NAME of \mem_reg_0_127_0_0__30\ : label is "inst/mem_reg";
  attribute RTL_RAM_TYPE of \mem_reg_0_127_0_0__30\ : label is "RAM_SP";
  attribute ram_addr_begin of \mem_reg_0_127_0_0__30\ : label is 0;
  attribute ram_addr_end of \mem_reg_0_127_0_0__30\ : label is 127;
  attribute ram_offset of \mem_reg_0_127_0_0__30\ : label is 0;
  attribute ram_slice_begin of \mem_reg_0_127_0_0__30\ : label is 31;
  attribute ram_slice_end of \mem_reg_0_127_0_0__30\ : label is 31;
  attribute RTL_RAM_BITS of \mem_reg_0_127_0_0__4\ : label is 4128;
  attribute RTL_RAM_NAME of \mem_reg_0_127_0_0__4\ : label is "inst/mem_reg";
  attribute RTL_RAM_TYPE of \mem_reg_0_127_0_0__4\ : label is "RAM_SP";
  attribute ram_addr_begin of \mem_reg_0_127_0_0__4\ : label is 0;
  attribute ram_addr_end of \mem_reg_0_127_0_0__4\ : label is 127;
  attribute ram_offset of \mem_reg_0_127_0_0__4\ : label is 0;
  attribute ram_slice_begin of \mem_reg_0_127_0_0__4\ : label is 5;
  attribute ram_slice_end of \mem_reg_0_127_0_0__4\ : label is 5;
  attribute RTL_RAM_BITS of \mem_reg_0_127_0_0__5\ : label is 4128;
  attribute RTL_RAM_NAME of \mem_reg_0_127_0_0__5\ : label is "inst/mem_reg";
  attribute RTL_RAM_TYPE of \mem_reg_0_127_0_0__5\ : label is "RAM_SP";
  attribute ram_addr_begin of \mem_reg_0_127_0_0__5\ : label is 0;
  attribute ram_addr_end of \mem_reg_0_127_0_0__5\ : label is 127;
  attribute ram_offset of \mem_reg_0_127_0_0__5\ : label is 0;
  attribute ram_slice_begin of \mem_reg_0_127_0_0__5\ : label is 6;
  attribute ram_slice_end of \mem_reg_0_127_0_0__5\ : label is 6;
  attribute RTL_RAM_BITS of \mem_reg_0_127_0_0__6\ : label is 4128;
  attribute RTL_RAM_NAME of \mem_reg_0_127_0_0__6\ : label is "inst/mem_reg";
  attribute RTL_RAM_TYPE of \mem_reg_0_127_0_0__6\ : label is "RAM_SP";
  attribute ram_addr_begin of \mem_reg_0_127_0_0__6\ : label is 0;
  attribute ram_addr_end of \mem_reg_0_127_0_0__6\ : label is 127;
  attribute ram_offset of \mem_reg_0_127_0_0__6\ : label is 0;
  attribute ram_slice_begin of \mem_reg_0_127_0_0__6\ : label is 7;
  attribute ram_slice_end of \mem_reg_0_127_0_0__6\ : label is 7;
  attribute RTL_RAM_BITS of \mem_reg_0_127_0_0__7\ : label is 4128;
  attribute RTL_RAM_NAME of \mem_reg_0_127_0_0__7\ : label is "inst/mem_reg";
  attribute RTL_RAM_TYPE of \mem_reg_0_127_0_0__7\ : label is "RAM_SP";
  attribute ram_addr_begin of \mem_reg_0_127_0_0__7\ : label is 0;
  attribute ram_addr_end of \mem_reg_0_127_0_0__7\ : label is 127;
  attribute ram_offset of \mem_reg_0_127_0_0__7\ : label is 0;
  attribute ram_slice_begin of \mem_reg_0_127_0_0__7\ : label is 8;
  attribute ram_slice_end of \mem_reg_0_127_0_0__7\ : label is 8;
  attribute RTL_RAM_BITS of \mem_reg_0_127_0_0__8\ : label is 4128;
  attribute RTL_RAM_NAME of \mem_reg_0_127_0_0__8\ : label is "inst/mem_reg";
  attribute RTL_RAM_TYPE of \mem_reg_0_127_0_0__8\ : label is "RAM_SP";
  attribute ram_addr_begin of \mem_reg_0_127_0_0__8\ : label is 0;
  attribute ram_addr_end of \mem_reg_0_127_0_0__8\ : label is 127;
  attribute ram_offset of \mem_reg_0_127_0_0__8\ : label is 0;
  attribute ram_slice_begin of \mem_reg_0_127_0_0__8\ : label is 9;
  attribute ram_slice_end of \mem_reg_0_127_0_0__8\ : label is 9;
  attribute RTL_RAM_BITS of \mem_reg_0_127_0_0__9\ : label is 4128;
  attribute RTL_RAM_NAME of \mem_reg_0_127_0_0__9\ : label is "inst/mem_reg";
  attribute RTL_RAM_TYPE of \mem_reg_0_127_0_0__9\ : label is "RAM_SP";
  attribute ram_addr_begin of \mem_reg_0_127_0_0__9\ : label is 0;
  attribute ram_addr_end of \mem_reg_0_127_0_0__9\ : label is 127;
  attribute ram_offset of \mem_reg_0_127_0_0__9\ : label is 0;
  attribute ram_slice_begin of \mem_reg_0_127_0_0__9\ : label is 10;
  attribute ram_slice_end of \mem_reg_0_127_0_0__9\ : label is 10;
  attribute RTL_RAM_BITS of mem_reg_0_15_0_0 : label is 4128;
  attribute RTL_RAM_NAME of mem_reg_0_15_0_0 : label is "inst/mem_reg";
  attribute RTL_RAM_TYPE of mem_reg_0_15_0_0 : label is "RAM_SP";
  attribute XILINX_LEGACY_PRIM : string;
  attribute XILINX_LEGACY_PRIM of mem_reg_0_15_0_0 : label is "RAM16X1S";
  attribute XILINX_TRANSFORM_PINMAP : string;
  attribute XILINX_TRANSFORM_PINMAP of mem_reg_0_15_0_0 : label is "GND:A4";
  attribute ram_addr_begin of mem_reg_0_15_0_0 : label is 128;
  attribute ram_addr_end of mem_reg_0_15_0_0 : label is 128;
  attribute ram_offset of mem_reg_0_15_0_0 : label is 0;
  attribute ram_slice_begin of mem_reg_0_15_0_0 : label is 0;
  attribute ram_slice_end of mem_reg_0_15_0_0 : label is 0;
  attribute RTL_RAM_BITS of \mem_reg_0_15_0_0__0\ : label is 4128;
  attribute RTL_RAM_NAME of \mem_reg_0_15_0_0__0\ : label is "inst/mem_reg";
  attribute RTL_RAM_TYPE of \mem_reg_0_15_0_0__0\ : label is "RAM_SP";
  attribute XILINX_LEGACY_PRIM of \mem_reg_0_15_0_0__0\ : label is "RAM16X1S";
  attribute XILINX_TRANSFORM_PINMAP of \mem_reg_0_15_0_0__0\ : label is "GND:A4";
  attribute ram_addr_begin of \mem_reg_0_15_0_0__0\ : label is 128;
  attribute ram_addr_end of \mem_reg_0_15_0_0__0\ : label is 128;
  attribute ram_offset of \mem_reg_0_15_0_0__0\ : label is 0;
  attribute ram_slice_begin of \mem_reg_0_15_0_0__0\ : label is 1;
  attribute ram_slice_end of \mem_reg_0_15_0_0__0\ : label is 1;
  attribute RTL_RAM_BITS of \mem_reg_0_15_0_0__1\ : label is 4128;
  attribute RTL_RAM_NAME of \mem_reg_0_15_0_0__1\ : label is "inst/mem_reg";
  attribute RTL_RAM_TYPE of \mem_reg_0_15_0_0__1\ : label is "RAM_SP";
  attribute XILINX_LEGACY_PRIM of \mem_reg_0_15_0_0__1\ : label is "RAM16X1S";
  attribute XILINX_TRANSFORM_PINMAP of \mem_reg_0_15_0_0__1\ : label is "GND:A4";
  attribute ram_addr_begin of \mem_reg_0_15_0_0__1\ : label is 128;
  attribute ram_addr_end of \mem_reg_0_15_0_0__1\ : label is 128;
  attribute ram_offset of \mem_reg_0_15_0_0__1\ : label is 0;
  attribute ram_slice_begin of \mem_reg_0_15_0_0__1\ : label is 2;
  attribute ram_slice_end of \mem_reg_0_15_0_0__1\ : label is 2;
  attribute RTL_RAM_BITS of \mem_reg_0_15_0_0__10\ : label is 4128;
  attribute RTL_RAM_NAME of \mem_reg_0_15_0_0__10\ : label is "inst/mem_reg";
  attribute RTL_RAM_TYPE of \mem_reg_0_15_0_0__10\ : label is "RAM_SP";
  attribute XILINX_LEGACY_PRIM of \mem_reg_0_15_0_0__10\ : label is "RAM16X1S";
  attribute XILINX_TRANSFORM_PINMAP of \mem_reg_0_15_0_0__10\ : label is "GND:A4";
  attribute ram_addr_begin of \mem_reg_0_15_0_0__10\ : label is 128;
  attribute ram_addr_end of \mem_reg_0_15_0_0__10\ : label is 128;
  attribute ram_offset of \mem_reg_0_15_0_0__10\ : label is 0;
  attribute ram_slice_begin of \mem_reg_0_15_0_0__10\ : label is 11;
  attribute ram_slice_end of \mem_reg_0_15_0_0__10\ : label is 11;
  attribute RTL_RAM_BITS of \mem_reg_0_15_0_0__11\ : label is 4128;
  attribute RTL_RAM_NAME of \mem_reg_0_15_0_0__11\ : label is "inst/mem_reg";
  attribute RTL_RAM_TYPE of \mem_reg_0_15_0_0__11\ : label is "RAM_SP";
  attribute XILINX_LEGACY_PRIM of \mem_reg_0_15_0_0__11\ : label is "RAM16X1S";
  attribute XILINX_TRANSFORM_PINMAP of \mem_reg_0_15_0_0__11\ : label is "GND:A4";
  attribute ram_addr_begin of \mem_reg_0_15_0_0__11\ : label is 128;
  attribute ram_addr_end of \mem_reg_0_15_0_0__11\ : label is 128;
  attribute ram_offset of \mem_reg_0_15_0_0__11\ : label is 0;
  attribute ram_slice_begin of \mem_reg_0_15_0_0__11\ : label is 12;
  attribute ram_slice_end of \mem_reg_0_15_0_0__11\ : label is 12;
  attribute RTL_RAM_BITS of \mem_reg_0_15_0_0__12\ : label is 4128;
  attribute RTL_RAM_NAME of \mem_reg_0_15_0_0__12\ : label is "inst/mem_reg";
  attribute RTL_RAM_TYPE of \mem_reg_0_15_0_0__12\ : label is "RAM_SP";
  attribute XILINX_LEGACY_PRIM of \mem_reg_0_15_0_0__12\ : label is "RAM16X1S";
  attribute XILINX_TRANSFORM_PINMAP of \mem_reg_0_15_0_0__12\ : label is "GND:A4";
  attribute ram_addr_begin of \mem_reg_0_15_0_0__12\ : label is 128;
  attribute ram_addr_end of \mem_reg_0_15_0_0__12\ : label is 128;
  attribute ram_offset of \mem_reg_0_15_0_0__12\ : label is 0;
  attribute ram_slice_begin of \mem_reg_0_15_0_0__12\ : label is 13;
  attribute ram_slice_end of \mem_reg_0_15_0_0__12\ : label is 13;
  attribute RTL_RAM_BITS of \mem_reg_0_15_0_0__13\ : label is 4128;
  attribute RTL_RAM_NAME of \mem_reg_0_15_0_0__13\ : label is "inst/mem_reg";
  attribute RTL_RAM_TYPE of \mem_reg_0_15_0_0__13\ : label is "RAM_SP";
  attribute XILINX_LEGACY_PRIM of \mem_reg_0_15_0_0__13\ : label is "RAM16X1S";
  attribute XILINX_TRANSFORM_PINMAP of \mem_reg_0_15_0_0__13\ : label is "GND:A4";
  attribute ram_addr_begin of \mem_reg_0_15_0_0__13\ : label is 128;
  attribute ram_addr_end of \mem_reg_0_15_0_0__13\ : label is 128;
  attribute ram_offset of \mem_reg_0_15_0_0__13\ : label is 0;
  attribute ram_slice_begin of \mem_reg_0_15_0_0__13\ : label is 14;
  attribute ram_slice_end of \mem_reg_0_15_0_0__13\ : label is 14;
  attribute RTL_RAM_BITS of \mem_reg_0_15_0_0__14\ : label is 4128;
  attribute RTL_RAM_NAME of \mem_reg_0_15_0_0__14\ : label is "inst/mem_reg";
  attribute RTL_RAM_TYPE of \mem_reg_0_15_0_0__14\ : label is "RAM_SP";
  attribute XILINX_LEGACY_PRIM of \mem_reg_0_15_0_0__14\ : label is "RAM16X1S";
  attribute XILINX_TRANSFORM_PINMAP of \mem_reg_0_15_0_0__14\ : label is "GND:A4";
  attribute ram_addr_begin of \mem_reg_0_15_0_0__14\ : label is 128;
  attribute ram_addr_end of \mem_reg_0_15_0_0__14\ : label is 128;
  attribute ram_offset of \mem_reg_0_15_0_0__14\ : label is 0;
  attribute ram_slice_begin of \mem_reg_0_15_0_0__14\ : label is 15;
  attribute ram_slice_end of \mem_reg_0_15_0_0__14\ : label is 15;
  attribute RTL_RAM_BITS of \mem_reg_0_15_0_0__15\ : label is 4128;
  attribute RTL_RAM_NAME of \mem_reg_0_15_0_0__15\ : label is "inst/mem_reg";
  attribute RTL_RAM_TYPE of \mem_reg_0_15_0_0__15\ : label is "RAM_SP";
  attribute XILINX_LEGACY_PRIM of \mem_reg_0_15_0_0__15\ : label is "RAM16X1S";
  attribute XILINX_TRANSFORM_PINMAP of \mem_reg_0_15_0_0__15\ : label is "GND:A4";
  attribute ram_addr_begin of \mem_reg_0_15_0_0__15\ : label is 128;
  attribute ram_addr_end of \mem_reg_0_15_0_0__15\ : label is 128;
  attribute ram_offset of \mem_reg_0_15_0_0__15\ : label is 0;
  attribute ram_slice_begin of \mem_reg_0_15_0_0__15\ : label is 16;
  attribute ram_slice_end of \mem_reg_0_15_0_0__15\ : label is 16;
  attribute RTL_RAM_BITS of \mem_reg_0_15_0_0__16\ : label is 4128;
  attribute RTL_RAM_NAME of \mem_reg_0_15_0_0__16\ : label is "inst/mem_reg";
  attribute RTL_RAM_TYPE of \mem_reg_0_15_0_0__16\ : label is "RAM_SP";
  attribute XILINX_LEGACY_PRIM of \mem_reg_0_15_0_0__16\ : label is "RAM16X1S";
  attribute XILINX_TRANSFORM_PINMAP of \mem_reg_0_15_0_0__16\ : label is "GND:A4";
  attribute ram_addr_begin of \mem_reg_0_15_0_0__16\ : label is 128;
  attribute ram_addr_end of \mem_reg_0_15_0_0__16\ : label is 128;
  attribute ram_offset of \mem_reg_0_15_0_0__16\ : label is 0;
  attribute ram_slice_begin of \mem_reg_0_15_0_0__16\ : label is 17;
  attribute ram_slice_end of \mem_reg_0_15_0_0__16\ : label is 17;
  attribute RTL_RAM_BITS of \mem_reg_0_15_0_0__17\ : label is 4128;
  attribute RTL_RAM_NAME of \mem_reg_0_15_0_0__17\ : label is "inst/mem_reg";
  attribute RTL_RAM_TYPE of \mem_reg_0_15_0_0__17\ : label is "RAM_SP";
  attribute XILINX_LEGACY_PRIM of \mem_reg_0_15_0_0__17\ : label is "RAM16X1S";
  attribute XILINX_TRANSFORM_PINMAP of \mem_reg_0_15_0_0__17\ : label is "GND:A4";
  attribute ram_addr_begin of \mem_reg_0_15_0_0__17\ : label is 128;
  attribute ram_addr_end of \mem_reg_0_15_0_0__17\ : label is 128;
  attribute ram_offset of \mem_reg_0_15_0_0__17\ : label is 0;
  attribute ram_slice_begin of \mem_reg_0_15_0_0__17\ : label is 18;
  attribute ram_slice_end of \mem_reg_0_15_0_0__17\ : label is 18;
  attribute RTL_RAM_BITS of \mem_reg_0_15_0_0__18\ : label is 4128;
  attribute RTL_RAM_NAME of \mem_reg_0_15_0_0__18\ : label is "inst/mem_reg";
  attribute RTL_RAM_TYPE of \mem_reg_0_15_0_0__18\ : label is "RAM_SP";
  attribute XILINX_LEGACY_PRIM of \mem_reg_0_15_0_0__18\ : label is "RAM16X1S";
  attribute XILINX_TRANSFORM_PINMAP of \mem_reg_0_15_0_0__18\ : label is "GND:A4";
  attribute ram_addr_begin of \mem_reg_0_15_0_0__18\ : label is 128;
  attribute ram_addr_end of \mem_reg_0_15_0_0__18\ : label is 128;
  attribute ram_offset of \mem_reg_0_15_0_0__18\ : label is 0;
  attribute ram_slice_begin of \mem_reg_0_15_0_0__18\ : label is 19;
  attribute ram_slice_end of \mem_reg_0_15_0_0__18\ : label is 19;
  attribute RTL_RAM_BITS of \mem_reg_0_15_0_0__19\ : label is 4128;
  attribute RTL_RAM_NAME of \mem_reg_0_15_0_0__19\ : label is "inst/mem_reg";
  attribute RTL_RAM_TYPE of \mem_reg_0_15_0_0__19\ : label is "RAM_SP";
  attribute XILINX_LEGACY_PRIM of \mem_reg_0_15_0_0__19\ : label is "RAM16X1S";
  attribute XILINX_TRANSFORM_PINMAP of \mem_reg_0_15_0_0__19\ : label is "GND:A4";
  attribute ram_addr_begin of \mem_reg_0_15_0_0__19\ : label is 128;
  attribute ram_addr_end of \mem_reg_0_15_0_0__19\ : label is 128;
  attribute ram_offset of \mem_reg_0_15_0_0__19\ : label is 0;
  attribute ram_slice_begin of \mem_reg_0_15_0_0__19\ : label is 20;
  attribute ram_slice_end of \mem_reg_0_15_0_0__19\ : label is 20;
  attribute RTL_RAM_BITS of \mem_reg_0_15_0_0__2\ : label is 4128;
  attribute RTL_RAM_NAME of \mem_reg_0_15_0_0__2\ : label is "inst/mem_reg";
  attribute RTL_RAM_TYPE of \mem_reg_0_15_0_0__2\ : label is "RAM_SP";
  attribute XILINX_LEGACY_PRIM of \mem_reg_0_15_0_0__2\ : label is "RAM16X1S";
  attribute XILINX_TRANSFORM_PINMAP of \mem_reg_0_15_0_0__2\ : label is "GND:A4";
  attribute ram_addr_begin of \mem_reg_0_15_0_0__2\ : label is 128;
  attribute ram_addr_end of \mem_reg_0_15_0_0__2\ : label is 128;
  attribute ram_offset of \mem_reg_0_15_0_0__2\ : label is 0;
  attribute ram_slice_begin of \mem_reg_0_15_0_0__2\ : label is 3;
  attribute ram_slice_end of \mem_reg_0_15_0_0__2\ : label is 3;
  attribute RTL_RAM_BITS of \mem_reg_0_15_0_0__20\ : label is 4128;
  attribute RTL_RAM_NAME of \mem_reg_0_15_0_0__20\ : label is "inst/mem_reg";
  attribute RTL_RAM_TYPE of \mem_reg_0_15_0_0__20\ : label is "RAM_SP";
  attribute XILINX_LEGACY_PRIM of \mem_reg_0_15_0_0__20\ : label is "RAM16X1S";
  attribute XILINX_TRANSFORM_PINMAP of \mem_reg_0_15_0_0__20\ : label is "GND:A4";
  attribute ram_addr_begin of \mem_reg_0_15_0_0__20\ : label is 128;
  attribute ram_addr_end of \mem_reg_0_15_0_0__20\ : label is 128;
  attribute ram_offset of \mem_reg_0_15_0_0__20\ : label is 0;
  attribute ram_slice_begin of \mem_reg_0_15_0_0__20\ : label is 21;
  attribute ram_slice_end of \mem_reg_0_15_0_0__20\ : label is 21;
  attribute RTL_RAM_BITS of \mem_reg_0_15_0_0__21\ : label is 4128;
  attribute RTL_RAM_NAME of \mem_reg_0_15_0_0__21\ : label is "inst/mem_reg";
  attribute RTL_RAM_TYPE of \mem_reg_0_15_0_0__21\ : label is "RAM_SP";
  attribute XILINX_LEGACY_PRIM of \mem_reg_0_15_0_0__21\ : label is "RAM16X1S";
  attribute XILINX_TRANSFORM_PINMAP of \mem_reg_0_15_0_0__21\ : label is "GND:A4";
  attribute ram_addr_begin of \mem_reg_0_15_0_0__21\ : label is 128;
  attribute ram_addr_end of \mem_reg_0_15_0_0__21\ : label is 128;
  attribute ram_offset of \mem_reg_0_15_0_0__21\ : label is 0;
  attribute ram_slice_begin of \mem_reg_0_15_0_0__21\ : label is 22;
  attribute ram_slice_end of \mem_reg_0_15_0_0__21\ : label is 22;
  attribute RTL_RAM_BITS of \mem_reg_0_15_0_0__22\ : label is 4128;
  attribute RTL_RAM_NAME of \mem_reg_0_15_0_0__22\ : label is "inst/mem_reg";
  attribute RTL_RAM_TYPE of \mem_reg_0_15_0_0__22\ : label is "RAM_SP";
  attribute XILINX_LEGACY_PRIM of \mem_reg_0_15_0_0__22\ : label is "RAM16X1S";
  attribute XILINX_TRANSFORM_PINMAP of \mem_reg_0_15_0_0__22\ : label is "GND:A4";
  attribute ram_addr_begin of \mem_reg_0_15_0_0__22\ : label is 128;
  attribute ram_addr_end of \mem_reg_0_15_0_0__22\ : label is 128;
  attribute ram_offset of \mem_reg_0_15_0_0__22\ : label is 0;
  attribute ram_slice_begin of \mem_reg_0_15_0_0__22\ : label is 23;
  attribute ram_slice_end of \mem_reg_0_15_0_0__22\ : label is 23;
  attribute RTL_RAM_BITS of \mem_reg_0_15_0_0__23\ : label is 4128;
  attribute RTL_RAM_NAME of \mem_reg_0_15_0_0__23\ : label is "inst/mem_reg";
  attribute RTL_RAM_TYPE of \mem_reg_0_15_0_0__23\ : label is "RAM_SP";
  attribute XILINX_LEGACY_PRIM of \mem_reg_0_15_0_0__23\ : label is "RAM16X1S";
  attribute XILINX_TRANSFORM_PINMAP of \mem_reg_0_15_0_0__23\ : label is "GND:A4";
  attribute ram_addr_begin of \mem_reg_0_15_0_0__23\ : label is 128;
  attribute ram_addr_end of \mem_reg_0_15_0_0__23\ : label is 128;
  attribute ram_offset of \mem_reg_0_15_0_0__23\ : label is 0;
  attribute ram_slice_begin of \mem_reg_0_15_0_0__23\ : label is 24;
  attribute ram_slice_end of \mem_reg_0_15_0_0__23\ : label is 24;
  attribute RTL_RAM_BITS of \mem_reg_0_15_0_0__24\ : label is 4128;
  attribute RTL_RAM_NAME of \mem_reg_0_15_0_0__24\ : label is "inst/mem_reg";
  attribute RTL_RAM_TYPE of \mem_reg_0_15_0_0__24\ : label is "RAM_SP";
  attribute XILINX_LEGACY_PRIM of \mem_reg_0_15_0_0__24\ : label is "RAM16X1S";
  attribute XILINX_TRANSFORM_PINMAP of \mem_reg_0_15_0_0__24\ : label is "GND:A4";
  attribute ram_addr_begin of \mem_reg_0_15_0_0__24\ : label is 128;
  attribute ram_addr_end of \mem_reg_0_15_0_0__24\ : label is 128;
  attribute ram_offset of \mem_reg_0_15_0_0__24\ : label is 0;
  attribute ram_slice_begin of \mem_reg_0_15_0_0__24\ : label is 25;
  attribute ram_slice_end of \mem_reg_0_15_0_0__24\ : label is 25;
  attribute RTL_RAM_BITS of \mem_reg_0_15_0_0__25\ : label is 4128;
  attribute RTL_RAM_NAME of \mem_reg_0_15_0_0__25\ : label is "inst/mem_reg";
  attribute RTL_RAM_TYPE of \mem_reg_0_15_0_0__25\ : label is "RAM_SP";
  attribute XILINX_LEGACY_PRIM of \mem_reg_0_15_0_0__25\ : label is "RAM16X1S";
  attribute XILINX_TRANSFORM_PINMAP of \mem_reg_0_15_0_0__25\ : label is "GND:A4";
  attribute ram_addr_begin of \mem_reg_0_15_0_0__25\ : label is 128;
  attribute ram_addr_end of \mem_reg_0_15_0_0__25\ : label is 128;
  attribute ram_offset of \mem_reg_0_15_0_0__25\ : label is 0;
  attribute ram_slice_begin of \mem_reg_0_15_0_0__25\ : label is 26;
  attribute ram_slice_end of \mem_reg_0_15_0_0__25\ : label is 26;
  attribute RTL_RAM_BITS of \mem_reg_0_15_0_0__26\ : label is 4128;
  attribute RTL_RAM_NAME of \mem_reg_0_15_0_0__26\ : label is "inst/mem_reg";
  attribute RTL_RAM_TYPE of \mem_reg_0_15_0_0__26\ : label is "RAM_SP";
  attribute XILINX_LEGACY_PRIM of \mem_reg_0_15_0_0__26\ : label is "RAM16X1S";
  attribute XILINX_TRANSFORM_PINMAP of \mem_reg_0_15_0_0__26\ : label is "GND:A4";
  attribute ram_addr_begin of \mem_reg_0_15_0_0__26\ : label is 128;
  attribute ram_addr_end of \mem_reg_0_15_0_0__26\ : label is 128;
  attribute ram_offset of \mem_reg_0_15_0_0__26\ : label is 0;
  attribute ram_slice_begin of \mem_reg_0_15_0_0__26\ : label is 27;
  attribute ram_slice_end of \mem_reg_0_15_0_0__26\ : label is 27;
  attribute RTL_RAM_BITS of \mem_reg_0_15_0_0__27\ : label is 4128;
  attribute RTL_RAM_NAME of \mem_reg_0_15_0_0__27\ : label is "inst/mem_reg";
  attribute RTL_RAM_TYPE of \mem_reg_0_15_0_0__27\ : label is "RAM_SP";
  attribute XILINX_LEGACY_PRIM of \mem_reg_0_15_0_0__27\ : label is "RAM16X1S";
  attribute XILINX_TRANSFORM_PINMAP of \mem_reg_0_15_0_0__27\ : label is "GND:A4";
  attribute ram_addr_begin of \mem_reg_0_15_0_0__27\ : label is 128;
  attribute ram_addr_end of \mem_reg_0_15_0_0__27\ : label is 128;
  attribute ram_offset of \mem_reg_0_15_0_0__27\ : label is 0;
  attribute ram_slice_begin of \mem_reg_0_15_0_0__27\ : label is 28;
  attribute ram_slice_end of \mem_reg_0_15_0_0__27\ : label is 28;
  attribute RTL_RAM_BITS of \mem_reg_0_15_0_0__28\ : label is 4128;
  attribute RTL_RAM_NAME of \mem_reg_0_15_0_0__28\ : label is "inst/mem_reg";
  attribute RTL_RAM_TYPE of \mem_reg_0_15_0_0__28\ : label is "RAM_SP";
  attribute XILINX_LEGACY_PRIM of \mem_reg_0_15_0_0__28\ : label is "RAM16X1S";
  attribute XILINX_TRANSFORM_PINMAP of \mem_reg_0_15_0_0__28\ : label is "GND:A4";
  attribute ram_addr_begin of \mem_reg_0_15_0_0__28\ : label is 128;
  attribute ram_addr_end of \mem_reg_0_15_0_0__28\ : label is 128;
  attribute ram_offset of \mem_reg_0_15_0_0__28\ : label is 0;
  attribute ram_slice_begin of \mem_reg_0_15_0_0__28\ : label is 29;
  attribute ram_slice_end of \mem_reg_0_15_0_0__28\ : label is 29;
  attribute RTL_RAM_BITS of \mem_reg_0_15_0_0__29\ : label is 4128;
  attribute RTL_RAM_NAME of \mem_reg_0_15_0_0__29\ : label is "inst/mem_reg";
  attribute RTL_RAM_TYPE of \mem_reg_0_15_0_0__29\ : label is "RAM_SP";
  attribute XILINX_LEGACY_PRIM of \mem_reg_0_15_0_0__29\ : label is "RAM16X1S";
  attribute XILINX_TRANSFORM_PINMAP of \mem_reg_0_15_0_0__29\ : label is "GND:A4";
  attribute ram_addr_begin of \mem_reg_0_15_0_0__29\ : label is 128;
  attribute ram_addr_end of \mem_reg_0_15_0_0__29\ : label is 128;
  attribute ram_offset of \mem_reg_0_15_0_0__29\ : label is 0;
  attribute ram_slice_begin of \mem_reg_0_15_0_0__29\ : label is 30;
  attribute ram_slice_end of \mem_reg_0_15_0_0__29\ : label is 30;
  attribute RTL_RAM_BITS of \mem_reg_0_15_0_0__3\ : label is 4128;
  attribute RTL_RAM_NAME of \mem_reg_0_15_0_0__3\ : label is "inst/mem_reg";
  attribute RTL_RAM_TYPE of \mem_reg_0_15_0_0__3\ : label is "RAM_SP";
  attribute XILINX_LEGACY_PRIM of \mem_reg_0_15_0_0__3\ : label is "RAM16X1S";
  attribute XILINX_TRANSFORM_PINMAP of \mem_reg_0_15_0_0__3\ : label is "GND:A4";
  attribute ram_addr_begin of \mem_reg_0_15_0_0__3\ : label is 128;
  attribute ram_addr_end of \mem_reg_0_15_0_0__3\ : label is 128;
  attribute ram_offset of \mem_reg_0_15_0_0__3\ : label is 0;
  attribute ram_slice_begin of \mem_reg_0_15_0_0__3\ : label is 4;
  attribute ram_slice_end of \mem_reg_0_15_0_0__3\ : label is 4;
  attribute RTL_RAM_BITS of \mem_reg_0_15_0_0__30\ : label is 4128;
  attribute RTL_RAM_NAME of \mem_reg_0_15_0_0__30\ : label is "inst/mem_reg";
  attribute RTL_RAM_TYPE of \mem_reg_0_15_0_0__30\ : label is "RAM_SP";
  attribute XILINX_LEGACY_PRIM of \mem_reg_0_15_0_0__30\ : label is "RAM16X1S";
  attribute XILINX_TRANSFORM_PINMAP of \mem_reg_0_15_0_0__30\ : label is "GND:A4";
  attribute ram_addr_begin of \mem_reg_0_15_0_0__30\ : label is 128;
  attribute ram_addr_end of \mem_reg_0_15_0_0__30\ : label is 128;
  attribute ram_offset of \mem_reg_0_15_0_0__30\ : label is 0;
  attribute ram_slice_begin of \mem_reg_0_15_0_0__30\ : label is 31;
  attribute ram_slice_end of \mem_reg_0_15_0_0__30\ : label is 31;
  attribute RTL_RAM_BITS of \mem_reg_0_15_0_0__4\ : label is 4128;
  attribute RTL_RAM_NAME of \mem_reg_0_15_0_0__4\ : label is "inst/mem_reg";
  attribute RTL_RAM_TYPE of \mem_reg_0_15_0_0__4\ : label is "RAM_SP";
  attribute XILINX_LEGACY_PRIM of \mem_reg_0_15_0_0__4\ : label is "RAM16X1S";
  attribute XILINX_TRANSFORM_PINMAP of \mem_reg_0_15_0_0__4\ : label is "GND:A4";
  attribute ram_addr_begin of \mem_reg_0_15_0_0__4\ : label is 128;
  attribute ram_addr_end of \mem_reg_0_15_0_0__4\ : label is 128;
  attribute ram_offset of \mem_reg_0_15_0_0__4\ : label is 0;
  attribute ram_slice_begin of \mem_reg_0_15_0_0__4\ : label is 5;
  attribute ram_slice_end of \mem_reg_0_15_0_0__4\ : label is 5;
  attribute RTL_RAM_BITS of \mem_reg_0_15_0_0__5\ : label is 4128;
  attribute RTL_RAM_NAME of \mem_reg_0_15_0_0__5\ : label is "inst/mem_reg";
  attribute RTL_RAM_TYPE of \mem_reg_0_15_0_0__5\ : label is "RAM_SP";
  attribute XILINX_LEGACY_PRIM of \mem_reg_0_15_0_0__5\ : label is "RAM16X1S";
  attribute XILINX_TRANSFORM_PINMAP of \mem_reg_0_15_0_0__5\ : label is "GND:A4";
  attribute ram_addr_begin of \mem_reg_0_15_0_0__5\ : label is 128;
  attribute ram_addr_end of \mem_reg_0_15_0_0__5\ : label is 128;
  attribute ram_offset of \mem_reg_0_15_0_0__5\ : label is 0;
  attribute ram_slice_begin of \mem_reg_0_15_0_0__5\ : label is 6;
  attribute ram_slice_end of \mem_reg_0_15_0_0__5\ : label is 6;
  attribute RTL_RAM_BITS of \mem_reg_0_15_0_0__6\ : label is 4128;
  attribute RTL_RAM_NAME of \mem_reg_0_15_0_0__6\ : label is "inst/mem_reg";
  attribute RTL_RAM_TYPE of \mem_reg_0_15_0_0__6\ : label is "RAM_SP";
  attribute XILINX_LEGACY_PRIM of \mem_reg_0_15_0_0__6\ : label is "RAM16X1S";
  attribute XILINX_TRANSFORM_PINMAP of \mem_reg_0_15_0_0__6\ : label is "GND:A4";
  attribute ram_addr_begin of \mem_reg_0_15_0_0__6\ : label is 128;
  attribute ram_addr_end of \mem_reg_0_15_0_0__6\ : label is 128;
  attribute ram_offset of \mem_reg_0_15_0_0__6\ : label is 0;
  attribute ram_slice_begin of \mem_reg_0_15_0_0__6\ : label is 7;
  attribute ram_slice_end of \mem_reg_0_15_0_0__6\ : label is 7;
  attribute RTL_RAM_BITS of \mem_reg_0_15_0_0__7\ : label is 4128;
  attribute RTL_RAM_NAME of \mem_reg_0_15_0_0__7\ : label is "inst/mem_reg";
  attribute RTL_RAM_TYPE of \mem_reg_0_15_0_0__7\ : label is "RAM_SP";
  attribute XILINX_LEGACY_PRIM of \mem_reg_0_15_0_0__7\ : label is "RAM16X1S";
  attribute XILINX_TRANSFORM_PINMAP of \mem_reg_0_15_0_0__7\ : label is "GND:A4";
  attribute ram_addr_begin of \mem_reg_0_15_0_0__7\ : label is 128;
  attribute ram_addr_end of \mem_reg_0_15_0_0__7\ : label is 128;
  attribute ram_offset of \mem_reg_0_15_0_0__7\ : label is 0;
  attribute ram_slice_begin of \mem_reg_0_15_0_0__7\ : label is 8;
  attribute ram_slice_end of \mem_reg_0_15_0_0__7\ : label is 8;
  attribute RTL_RAM_BITS of \mem_reg_0_15_0_0__8\ : label is 4128;
  attribute RTL_RAM_NAME of \mem_reg_0_15_0_0__8\ : label is "inst/mem_reg";
  attribute RTL_RAM_TYPE of \mem_reg_0_15_0_0__8\ : label is "RAM_SP";
  attribute XILINX_LEGACY_PRIM of \mem_reg_0_15_0_0__8\ : label is "RAM16X1S";
  attribute XILINX_TRANSFORM_PINMAP of \mem_reg_0_15_0_0__8\ : label is "GND:A4";
  attribute ram_addr_begin of \mem_reg_0_15_0_0__8\ : label is 128;
  attribute ram_addr_end of \mem_reg_0_15_0_0__8\ : label is 128;
  attribute ram_offset of \mem_reg_0_15_0_0__8\ : label is 0;
  attribute ram_slice_begin of \mem_reg_0_15_0_0__8\ : label is 9;
  attribute ram_slice_end of \mem_reg_0_15_0_0__8\ : label is 9;
  attribute RTL_RAM_BITS of \mem_reg_0_15_0_0__9\ : label is 4128;
  attribute RTL_RAM_NAME of \mem_reg_0_15_0_0__9\ : label is "inst/mem_reg";
  attribute RTL_RAM_TYPE of \mem_reg_0_15_0_0__9\ : label is "RAM_SP";
  attribute XILINX_LEGACY_PRIM of \mem_reg_0_15_0_0__9\ : label is "RAM16X1S";
  attribute XILINX_TRANSFORM_PINMAP of \mem_reg_0_15_0_0__9\ : label is "GND:A4";
  attribute ram_addr_begin of \mem_reg_0_15_0_0__9\ : label is 128;
  attribute ram_addr_end of \mem_reg_0_15_0_0__9\ : label is 128;
  attribute ram_offset of \mem_reg_0_15_0_0__9\ : label is 0;
  attribute ram_slice_begin of \mem_reg_0_15_0_0__9\ : label is 10;
  attribute ram_slice_end of \mem_reg_0_15_0_0__9\ : label is 10;
begin
\d_out[0]_INST_0\: unisim.vcomponents.LUT6
    generic map(
      INIT => X"0004FFFF00040000"
    )
        port map (
      I0 => addr(5),
      I1 => mem_reg_0_15_0_0_n_0,
      I2 => addr(4),
      I3 => addr(6),
      I4 => addr(7),
      I5 => mem_reg_0_127_0_0_n_0,
      O => d_out(0)
    );
\d_out[10]_INST_0\: unisim.vcomponents.LUT6
    generic map(
      INIT => X"0004FFFF00040000"
    )
        port map (
      I0 => addr(5),
      I1 => \mem_reg_0_15_0_0__9_n_0\,
      I2 => addr(4),
      I3 => addr(6),
      I4 => addr(7),
      I5 => \mem_reg_0_127_0_0__9_n_0\,
      O => d_out(10)
    );
\d_out[11]_INST_0\: unisim.vcomponents.LUT6
    generic map(
      INIT => X"0004FFFF00040000"
    )
        port map (
      I0 => addr(5),
      I1 => \mem_reg_0_15_0_0__10_n_0\,
      I2 => addr(4),
      I3 => addr(6),
      I4 => addr(7),
      I5 => \mem_reg_0_127_0_0__10_n_0\,
      O => d_out(11)
    );
\d_out[12]_INST_0\: unisim.vcomponents.LUT6
    generic map(
      INIT => X"0004FFFF00040000"
    )
        port map (
      I0 => addr(5),
      I1 => \mem_reg_0_15_0_0__11_n_0\,
      I2 => addr(4),
      I3 => addr(6),
      I4 => addr(7),
      I5 => \mem_reg_0_127_0_0__11_n_0\,
      O => d_out(12)
    );
\d_out[13]_INST_0\: unisim.vcomponents.LUT6
    generic map(
      INIT => X"0004FFFF00040000"
    )
        port map (
      I0 => addr(5),
      I1 => \mem_reg_0_15_0_0__12_n_0\,
      I2 => addr(4),
      I3 => addr(6),
      I4 => addr(7),
      I5 => \mem_reg_0_127_0_0__12_n_0\,
      O => d_out(13)
    );
\d_out[14]_INST_0\: unisim.vcomponents.LUT6
    generic map(
      INIT => X"0004FFFF00040000"
    )
        port map (
      I0 => addr(5),
      I1 => \mem_reg_0_15_0_0__13_n_0\,
      I2 => addr(4),
      I3 => addr(6),
      I4 => addr(7),
      I5 => \mem_reg_0_127_0_0__13_n_0\,
      O => d_out(14)
    );
\d_out[15]_INST_0\: unisim.vcomponents.LUT6
    generic map(
      INIT => X"0004FFFF00040000"
    )
        port map (
      I0 => addr(5),
      I1 => \mem_reg_0_15_0_0__14_n_0\,
      I2 => addr(4),
      I3 => addr(6),
      I4 => addr(7),
      I5 => \mem_reg_0_127_0_0__14_n_0\,
      O => d_out(15)
    );
\d_out[16]_INST_0\: unisim.vcomponents.LUT6
    generic map(
      INIT => X"0004FFFF00040000"
    )
        port map (
      I0 => addr(5),
      I1 => \mem_reg_0_15_0_0__15_n_0\,
      I2 => addr(4),
      I3 => addr(6),
      I4 => addr(7),
      I5 => \mem_reg_0_127_0_0__15_n_0\,
      O => d_out(16)
    );
\d_out[17]_INST_0\: unisim.vcomponents.LUT6
    generic map(
      INIT => X"0004FFFF00040000"
    )
        port map (
      I0 => addr(5),
      I1 => \mem_reg_0_15_0_0__16_n_0\,
      I2 => addr(4),
      I3 => addr(6),
      I4 => addr(7),
      I5 => \mem_reg_0_127_0_0__16_n_0\,
      O => d_out(17)
    );
\d_out[18]_INST_0\: unisim.vcomponents.LUT6
    generic map(
      INIT => X"0004FFFF00040000"
    )
        port map (
      I0 => addr(5),
      I1 => \mem_reg_0_15_0_0__17_n_0\,
      I2 => addr(4),
      I3 => addr(6),
      I4 => addr(7),
      I5 => \mem_reg_0_127_0_0__17_n_0\,
      O => d_out(18)
    );
\d_out[19]_INST_0\: unisim.vcomponents.LUT6
    generic map(
      INIT => X"0004FFFF00040000"
    )
        port map (
      I0 => addr(5),
      I1 => \mem_reg_0_15_0_0__18_n_0\,
      I2 => addr(4),
      I3 => addr(6),
      I4 => addr(7),
      I5 => \mem_reg_0_127_0_0__18_n_0\,
      O => d_out(19)
    );
\d_out[1]_INST_0\: unisim.vcomponents.LUT6
    generic map(
      INIT => X"0004FFFF00040000"
    )
        port map (
      I0 => addr(5),
      I1 => \mem_reg_0_15_0_0__0_n_0\,
      I2 => addr(4),
      I3 => addr(6),
      I4 => addr(7),
      I5 => \mem_reg_0_127_0_0__0_n_0\,
      O => d_out(1)
    );
\d_out[20]_INST_0\: unisim.vcomponents.LUT6
    generic map(
      INIT => X"0004FFFF00040000"
    )
        port map (
      I0 => addr(5),
      I1 => \mem_reg_0_15_0_0__19_n_0\,
      I2 => addr(4),
      I3 => addr(6),
      I4 => addr(7),
      I5 => \mem_reg_0_127_0_0__19_n_0\,
      O => d_out(20)
    );
\d_out[21]_INST_0\: unisim.vcomponents.LUT6
    generic map(
      INIT => X"0004FFFF00040000"
    )
        port map (
      I0 => addr(5),
      I1 => \mem_reg_0_15_0_0__20_n_0\,
      I2 => addr(4),
      I3 => addr(6),
      I4 => addr(7),
      I5 => \mem_reg_0_127_0_0__20_n_0\,
      O => d_out(21)
    );
\d_out[22]_INST_0\: unisim.vcomponents.LUT6
    generic map(
      INIT => X"0004FFFF00040000"
    )
        port map (
      I0 => addr(5),
      I1 => \mem_reg_0_15_0_0__21_n_0\,
      I2 => addr(4),
      I3 => addr(6),
      I4 => addr(7),
      I5 => \mem_reg_0_127_0_0__21_n_0\,
      O => d_out(22)
    );
\d_out[23]_INST_0\: unisim.vcomponents.LUT6
    generic map(
      INIT => X"0004FFFF00040000"
    )
        port map (
      I0 => addr(5),
      I1 => \mem_reg_0_15_0_0__22_n_0\,
      I2 => addr(4),
      I3 => addr(6),
      I4 => addr(7),
      I5 => \mem_reg_0_127_0_0__22_n_0\,
      O => d_out(23)
    );
\d_out[24]_INST_0\: unisim.vcomponents.LUT6
    generic map(
      INIT => X"0004FFFF00040000"
    )
        port map (
      I0 => addr(5),
      I1 => \mem_reg_0_15_0_0__23_n_0\,
      I2 => addr(4),
      I3 => addr(6),
      I4 => addr(7),
      I5 => \mem_reg_0_127_0_0__23_n_0\,
      O => d_out(24)
    );
\d_out[25]_INST_0\: unisim.vcomponents.LUT6
    generic map(
      INIT => X"0004FFFF00040000"
    )
        port map (
      I0 => addr(5),
      I1 => \mem_reg_0_15_0_0__24_n_0\,
      I2 => addr(4),
      I3 => addr(6),
      I4 => addr(7),
      I5 => \mem_reg_0_127_0_0__24_n_0\,
      O => d_out(25)
    );
\d_out[26]_INST_0\: unisim.vcomponents.LUT6
    generic map(
      INIT => X"0004FFFF00040000"
    )
        port map (
      I0 => addr(5),
      I1 => \mem_reg_0_15_0_0__25_n_0\,
      I2 => addr(4),
      I3 => addr(6),
      I4 => addr(7),
      I5 => \mem_reg_0_127_0_0__25_n_0\,
      O => d_out(26)
    );
\d_out[27]_INST_0\: unisim.vcomponents.LUT6
    generic map(
      INIT => X"0004FFFF00040000"
    )
        port map (
      I0 => addr(5),
      I1 => \mem_reg_0_15_0_0__26_n_0\,
      I2 => addr(4),
      I3 => addr(6),
      I4 => addr(7),
      I5 => \mem_reg_0_127_0_0__26_n_0\,
      O => d_out(27)
    );
\d_out[28]_INST_0\: unisim.vcomponents.LUT6
    generic map(
      INIT => X"0004FFFF00040000"
    )
        port map (
      I0 => addr(5),
      I1 => \mem_reg_0_15_0_0__27_n_0\,
      I2 => addr(4),
      I3 => addr(6),
      I4 => addr(7),
      I5 => \mem_reg_0_127_0_0__27_n_0\,
      O => d_out(28)
    );
\d_out[29]_INST_0\: unisim.vcomponents.LUT6
    generic map(
      INIT => X"0004FFFF00040000"
    )
        port map (
      I0 => addr(5),
      I1 => \mem_reg_0_15_0_0__28_n_0\,
      I2 => addr(4),
      I3 => addr(6),
      I4 => addr(7),
      I5 => \mem_reg_0_127_0_0__28_n_0\,
      O => d_out(29)
    );
\d_out[2]_INST_0\: unisim.vcomponents.LUT6
    generic map(
      INIT => X"0004FFFF00040000"
    )
        port map (
      I0 => addr(5),
      I1 => \mem_reg_0_15_0_0__1_n_0\,
      I2 => addr(4),
      I3 => addr(6),
      I4 => addr(7),
      I5 => \mem_reg_0_127_0_0__1_n_0\,
      O => d_out(2)
    );
\d_out[30]_INST_0\: unisim.vcomponents.LUT6
    generic map(
      INIT => X"0004FFFF00040000"
    )
        port map (
      I0 => addr(5),
      I1 => \mem_reg_0_15_0_0__29_n_0\,
      I2 => addr(4),
      I3 => addr(6),
      I4 => addr(7),
      I5 => \mem_reg_0_127_0_0__29_n_0\,
      O => d_out(30)
    );
\d_out[31]_INST_0\: unisim.vcomponents.LUT6
    generic map(
      INIT => X"0004FFFF00040000"
    )
        port map (
      I0 => addr(5),
      I1 => \mem_reg_0_15_0_0__30_n_0\,
      I2 => addr(4),
      I3 => addr(6),
      I4 => addr(7),
      I5 => \mem_reg_0_127_0_0__30_n_0\,
      O => d_out(31)
    );
\d_out[3]_INST_0\: unisim.vcomponents.LUT6
    generic map(
      INIT => X"0004FFFF00040000"
    )
        port map (
      I0 => addr(5),
      I1 => \mem_reg_0_15_0_0__2_n_0\,
      I2 => addr(4),
      I3 => addr(6),
      I4 => addr(7),
      I5 => \mem_reg_0_127_0_0__2_n_0\,
      O => d_out(3)
    );
\d_out[4]_INST_0\: unisim.vcomponents.LUT6
    generic map(
      INIT => X"0004FFFF00040000"
    )
        port map (
      I0 => addr(5),
      I1 => \mem_reg_0_15_0_0__3_n_0\,
      I2 => addr(4),
      I3 => addr(6),
      I4 => addr(7),
      I5 => \mem_reg_0_127_0_0__3_n_0\,
      O => d_out(4)
    );
\d_out[5]_INST_0\: unisim.vcomponents.LUT6
    generic map(
      INIT => X"0004FFFF00040000"
    )
        port map (
      I0 => addr(5),
      I1 => \mem_reg_0_15_0_0__4_n_0\,
      I2 => addr(4),
      I3 => addr(6),
      I4 => addr(7),
      I5 => \mem_reg_0_127_0_0__4_n_0\,
      O => d_out(5)
    );
\d_out[6]_INST_0\: unisim.vcomponents.LUT6
    generic map(
      INIT => X"0004FFFF00040000"
    )
        port map (
      I0 => addr(5),
      I1 => \mem_reg_0_15_0_0__5_n_0\,
      I2 => addr(4),
      I3 => addr(6),
      I4 => addr(7),
      I5 => \mem_reg_0_127_0_0__5_n_0\,
      O => d_out(6)
    );
\d_out[7]_INST_0\: unisim.vcomponents.LUT6
    generic map(
      INIT => X"0004FFFF00040000"
    )
        port map (
      I0 => addr(5),
      I1 => \mem_reg_0_15_0_0__6_n_0\,
      I2 => addr(4),
      I3 => addr(6),
      I4 => addr(7),
      I5 => \mem_reg_0_127_0_0__6_n_0\,
      O => d_out(7)
    );
\d_out[8]_INST_0\: unisim.vcomponents.LUT6
    generic map(
      INIT => X"0004FFFF00040000"
    )
        port map (
      I0 => addr(5),
      I1 => \mem_reg_0_15_0_0__7_n_0\,
      I2 => addr(4),
      I3 => addr(6),
      I4 => addr(7),
      I5 => \mem_reg_0_127_0_0__7_n_0\,
      O => d_out(8)
    );
\d_out[9]_INST_0\: unisim.vcomponents.LUT6
    generic map(
      INIT => X"0004FFFF00040000"
    )
        port map (
      I0 => addr(5),
      I1 => \mem_reg_0_15_0_0__8_n_0\,
      I2 => addr(4),
      I3 => addr(6),
      I4 => addr(7),
      I5 => \mem_reg_0_127_0_0__8_n_0\,
      O => d_out(9)
    );
mem_reg_0_127_0_0: unisim.vcomponents.RAM128X1S
     port map (
      A0 => addr(0),
      A1 => addr(1),
      A2 => addr(2),
      A3 => addr(3),
      A4 => addr(4),
      A5 => addr(5),
      A6 => addr(6),
      D => d_in(0),
      O => mem_reg_0_127_0_0_n_0,
      WCLK => clk,
      WE => mem_reg_0_127_0_0_i_1_n_0
    );
\mem_reg_0_127_0_0__0\: unisim.vcomponents.RAM128X1S
     port map (
      A0 => addr(0),
      A1 => addr(1),
      A2 => addr(2),
      A3 => addr(3),
      A4 => addr(4),
      A5 => addr(5),
      A6 => addr(6),
      D => d_in(1),
      O => \mem_reg_0_127_0_0__0_n_0\,
      WCLK => clk,
      WE => mem_reg_0_127_0_0_i_1_n_0
    );
\mem_reg_0_127_0_0__1\: unisim.vcomponents.RAM128X1S
     port map (
      A0 => addr(0),
      A1 => addr(1),
      A2 => addr(2),
      A3 => addr(3),
      A4 => addr(4),
      A5 => addr(5),
      A6 => addr(6),
      D => d_in(2),
      O => \mem_reg_0_127_0_0__1_n_0\,
      WCLK => clk,
      WE => mem_reg_0_127_0_0_i_1_n_0
    );
\mem_reg_0_127_0_0__10\: unisim.vcomponents.RAM128X1S
     port map (
      A0 => addr(0),
      A1 => addr(1),
      A2 => addr(2),
      A3 => addr(3),
      A4 => addr(4),
      A5 => addr(5),
      A6 => addr(6),
      D => d_in(11),
      O => \mem_reg_0_127_0_0__10_n_0\,
      WCLK => clk,
      WE => mem_reg_0_127_0_0_i_1_n_0
    );
\mem_reg_0_127_0_0__11\: unisim.vcomponents.RAM128X1S
     port map (
      A0 => addr(0),
      A1 => addr(1),
      A2 => addr(2),
      A3 => addr(3),
      A4 => addr(4),
      A5 => addr(5),
      A6 => addr(6),
      D => d_in(12),
      O => \mem_reg_0_127_0_0__11_n_0\,
      WCLK => clk,
      WE => mem_reg_0_127_0_0_i_1_n_0
    );
\mem_reg_0_127_0_0__12\: unisim.vcomponents.RAM128X1S
     port map (
      A0 => addr(0),
      A1 => addr(1),
      A2 => addr(2),
      A3 => addr(3),
      A4 => addr(4),
      A5 => addr(5),
      A6 => addr(6),
      D => d_in(13),
      O => \mem_reg_0_127_0_0__12_n_0\,
      WCLK => clk,
      WE => mem_reg_0_127_0_0_i_1_n_0
    );
\mem_reg_0_127_0_0__13\: unisim.vcomponents.RAM128X1S
     port map (
      A0 => addr(0),
      A1 => addr(1),
      A2 => addr(2),
      A3 => addr(3),
      A4 => addr(4),
      A5 => addr(5),
      A6 => addr(6),
      D => d_in(14),
      O => \mem_reg_0_127_0_0__13_n_0\,
      WCLK => clk,
      WE => mem_reg_0_127_0_0_i_1_n_0
    );
\mem_reg_0_127_0_0__14\: unisim.vcomponents.RAM128X1S
     port map (
      A0 => addr(0),
      A1 => addr(1),
      A2 => addr(2),
      A3 => addr(3),
      A4 => addr(4),
      A5 => addr(5),
      A6 => addr(6),
      D => d_in(15),
      O => \mem_reg_0_127_0_0__14_n_0\,
      WCLK => clk,
      WE => mem_reg_0_127_0_0_i_1_n_0
    );
\mem_reg_0_127_0_0__15\: unisim.vcomponents.RAM128X1S
     port map (
      A0 => addr(0),
      A1 => addr(1),
      A2 => addr(2),
      A3 => addr(3),
      A4 => addr(4),
      A5 => addr(5),
      A6 => addr(6),
      D => d_in(16),
      O => \mem_reg_0_127_0_0__15_n_0\,
      WCLK => clk,
      WE => mem_reg_0_127_0_0_i_1_n_0
    );
\mem_reg_0_127_0_0__16\: unisim.vcomponents.RAM128X1S
     port map (
      A0 => addr(0),
      A1 => addr(1),
      A2 => addr(2),
      A3 => addr(3),
      A4 => addr(4),
      A5 => addr(5),
      A6 => addr(6),
      D => d_in(17),
      O => \mem_reg_0_127_0_0__16_n_0\,
      WCLK => clk,
      WE => mem_reg_0_127_0_0_i_1_n_0
    );
\mem_reg_0_127_0_0__17\: unisim.vcomponents.RAM128X1S
     port map (
      A0 => addr(0),
      A1 => addr(1),
      A2 => addr(2),
      A3 => addr(3),
      A4 => addr(4),
      A5 => addr(5),
      A6 => addr(6),
      D => d_in(18),
      O => \mem_reg_0_127_0_0__17_n_0\,
      WCLK => clk,
      WE => mem_reg_0_127_0_0_i_1_n_0
    );
\mem_reg_0_127_0_0__18\: unisim.vcomponents.RAM128X1S
     port map (
      A0 => addr(0),
      A1 => addr(1),
      A2 => addr(2),
      A3 => addr(3),
      A4 => addr(4),
      A5 => addr(5),
      A6 => addr(6),
      D => d_in(19),
      O => \mem_reg_0_127_0_0__18_n_0\,
      WCLK => clk,
      WE => mem_reg_0_127_0_0_i_1_n_0
    );
\mem_reg_0_127_0_0__19\: unisim.vcomponents.RAM128X1S
     port map (
      A0 => addr(0),
      A1 => addr(1),
      A2 => addr(2),
      A3 => addr(3),
      A4 => addr(4),
      A5 => addr(5),
      A6 => addr(6),
      D => d_in(20),
      O => \mem_reg_0_127_0_0__19_n_0\,
      WCLK => clk,
      WE => mem_reg_0_127_0_0_i_1_n_0
    );
\mem_reg_0_127_0_0__2\: unisim.vcomponents.RAM128X1S
     port map (
      A0 => addr(0),
      A1 => addr(1),
      A2 => addr(2),
      A3 => addr(3),
      A4 => addr(4),
      A5 => addr(5),
      A6 => addr(6),
      D => d_in(3),
      O => \mem_reg_0_127_0_0__2_n_0\,
      WCLK => clk,
      WE => mem_reg_0_127_0_0_i_1_n_0
    );
\mem_reg_0_127_0_0__20\: unisim.vcomponents.RAM128X1S
     port map (
      A0 => addr(0),
      A1 => addr(1),
      A2 => addr(2),
      A3 => addr(3),
      A4 => addr(4),
      A5 => addr(5),
      A6 => addr(6),
      D => d_in(21),
      O => \mem_reg_0_127_0_0__20_n_0\,
      WCLK => clk,
      WE => mem_reg_0_127_0_0_i_1_n_0
    );
\mem_reg_0_127_0_0__21\: unisim.vcomponents.RAM128X1S
     port map (
      A0 => addr(0),
      A1 => addr(1),
      A2 => addr(2),
      A3 => addr(3),
      A4 => addr(4),
      A5 => addr(5),
      A6 => addr(6),
      D => d_in(22),
      O => \mem_reg_0_127_0_0__21_n_0\,
      WCLK => clk,
      WE => mem_reg_0_127_0_0_i_1_n_0
    );
\mem_reg_0_127_0_0__22\: unisim.vcomponents.RAM128X1S
     port map (
      A0 => addr(0),
      A1 => addr(1),
      A2 => addr(2),
      A3 => addr(3),
      A4 => addr(4),
      A5 => addr(5),
      A6 => addr(6),
      D => d_in(23),
      O => \mem_reg_0_127_0_0__22_n_0\,
      WCLK => clk,
      WE => mem_reg_0_127_0_0_i_1_n_0
    );
\mem_reg_0_127_0_0__23\: unisim.vcomponents.RAM128X1S
     port map (
      A0 => addr(0),
      A1 => addr(1),
      A2 => addr(2),
      A3 => addr(3),
      A4 => addr(4),
      A5 => addr(5),
      A6 => addr(6),
      D => d_in(24),
      O => \mem_reg_0_127_0_0__23_n_0\,
      WCLK => clk,
      WE => mem_reg_0_127_0_0_i_1_n_0
    );
\mem_reg_0_127_0_0__24\: unisim.vcomponents.RAM128X1S
     port map (
      A0 => addr(0),
      A1 => addr(1),
      A2 => addr(2),
      A3 => addr(3),
      A4 => addr(4),
      A5 => addr(5),
      A6 => addr(6),
      D => d_in(25),
      O => \mem_reg_0_127_0_0__24_n_0\,
      WCLK => clk,
      WE => mem_reg_0_127_0_0_i_1_n_0
    );
\mem_reg_0_127_0_0__25\: unisim.vcomponents.RAM128X1S
     port map (
      A0 => addr(0),
      A1 => addr(1),
      A2 => addr(2),
      A3 => addr(3),
      A4 => addr(4),
      A5 => addr(5),
      A6 => addr(6),
      D => d_in(26),
      O => \mem_reg_0_127_0_0__25_n_0\,
      WCLK => clk,
      WE => mem_reg_0_127_0_0_i_1_n_0
    );
\mem_reg_0_127_0_0__26\: unisim.vcomponents.RAM128X1S
     port map (
      A0 => addr(0),
      A1 => addr(1),
      A2 => addr(2),
      A3 => addr(3),
      A4 => addr(4),
      A5 => addr(5),
      A6 => addr(6),
      D => d_in(27),
      O => \mem_reg_0_127_0_0__26_n_0\,
      WCLK => clk,
      WE => mem_reg_0_127_0_0_i_1_n_0
    );
\mem_reg_0_127_0_0__27\: unisim.vcomponents.RAM128X1S
     port map (
      A0 => addr(0),
      A1 => addr(1),
      A2 => addr(2),
      A3 => addr(3),
      A4 => addr(4),
      A5 => addr(5),
      A6 => addr(6),
      D => d_in(28),
      O => \mem_reg_0_127_0_0__27_n_0\,
      WCLK => clk,
      WE => mem_reg_0_127_0_0_i_1_n_0
    );
\mem_reg_0_127_0_0__28\: unisim.vcomponents.RAM128X1S
     port map (
      A0 => addr(0),
      A1 => addr(1),
      A2 => addr(2),
      A3 => addr(3),
      A4 => addr(4),
      A5 => addr(5),
      A6 => addr(6),
      D => d_in(29),
      O => \mem_reg_0_127_0_0__28_n_0\,
      WCLK => clk,
      WE => mem_reg_0_127_0_0_i_1_n_0
    );
\mem_reg_0_127_0_0__29\: unisim.vcomponents.RAM128X1S
     port map (
      A0 => addr(0),
      A1 => addr(1),
      A2 => addr(2),
      A3 => addr(3),
      A4 => addr(4),
      A5 => addr(5),
      A6 => addr(6),
      D => d_in(30),
      O => \mem_reg_0_127_0_0__29_n_0\,
      WCLK => clk,
      WE => mem_reg_0_127_0_0_i_1_n_0
    );
\mem_reg_0_127_0_0__3\: unisim.vcomponents.RAM128X1S
     port map (
      A0 => addr(0),
      A1 => addr(1),
      A2 => addr(2),
      A3 => addr(3),
      A4 => addr(4),
      A5 => addr(5),
      A6 => addr(6),
      D => d_in(4),
      O => \mem_reg_0_127_0_0__3_n_0\,
      WCLK => clk,
      WE => mem_reg_0_127_0_0_i_1_n_0
    );
\mem_reg_0_127_0_0__30\: unisim.vcomponents.RAM128X1S
     port map (
      A0 => addr(0),
      A1 => addr(1),
      A2 => addr(2),
      A3 => addr(3),
      A4 => addr(4),
      A5 => addr(5),
      A6 => addr(6),
      D => d_in(31),
      O => \mem_reg_0_127_0_0__30_n_0\,
      WCLK => clk,
      WE => mem_reg_0_127_0_0_i_1_n_0
    );
\mem_reg_0_127_0_0__4\: unisim.vcomponents.RAM128X1S
     port map (
      A0 => addr(0),
      A1 => addr(1),
      A2 => addr(2),
      A3 => addr(3),
      A4 => addr(4),
      A5 => addr(5),
      A6 => addr(6),
      D => d_in(5),
      O => \mem_reg_0_127_0_0__4_n_0\,
      WCLK => clk,
      WE => mem_reg_0_127_0_0_i_1_n_0
    );
\mem_reg_0_127_0_0__5\: unisim.vcomponents.RAM128X1S
     port map (
      A0 => addr(0),
      A1 => addr(1),
      A2 => addr(2),
      A3 => addr(3),
      A4 => addr(4),
      A5 => addr(5),
      A6 => addr(6),
      D => d_in(6),
      O => \mem_reg_0_127_0_0__5_n_0\,
      WCLK => clk,
      WE => mem_reg_0_127_0_0_i_1_n_0
    );
\mem_reg_0_127_0_0__6\: unisim.vcomponents.RAM128X1S
     port map (
      A0 => addr(0),
      A1 => addr(1),
      A2 => addr(2),
      A3 => addr(3),
      A4 => addr(4),
      A5 => addr(5),
      A6 => addr(6),
      D => d_in(7),
      O => \mem_reg_0_127_0_0__6_n_0\,
      WCLK => clk,
      WE => mem_reg_0_127_0_0_i_1_n_0
    );
\mem_reg_0_127_0_0__7\: unisim.vcomponents.RAM128X1S
     port map (
      A0 => addr(0),
      A1 => addr(1),
      A2 => addr(2),
      A3 => addr(3),
      A4 => addr(4),
      A5 => addr(5),
      A6 => addr(6),
      D => d_in(8),
      O => \mem_reg_0_127_0_0__7_n_0\,
      WCLK => clk,
      WE => mem_reg_0_127_0_0_i_1_n_0
    );
\mem_reg_0_127_0_0__8\: unisim.vcomponents.RAM128X1S
     port map (
      A0 => addr(0),
      A1 => addr(1),
      A2 => addr(2),
      A3 => addr(3),
      A4 => addr(4),
      A5 => addr(5),
      A6 => addr(6),
      D => d_in(9),
      O => \mem_reg_0_127_0_0__8_n_0\,
      WCLK => clk,
      WE => mem_reg_0_127_0_0_i_1_n_0
    );
\mem_reg_0_127_0_0__9\: unisim.vcomponents.RAM128X1S
     port map (
      A0 => addr(0),
      A1 => addr(1),
      A2 => addr(2),
      A3 => addr(3),
      A4 => addr(4),
      A5 => addr(5),
      A6 => addr(6),
      D => d_in(10),
      O => \mem_reg_0_127_0_0__9_n_0\,
      WCLK => clk,
      WE => mem_reg_0_127_0_0_i_1_n_0
    );
mem_reg_0_127_0_0_i_1: unisim.vcomponents.LUT2
    generic map(
      INIT => X"2"
    )
        port map (
      I0 => w_ena,
      I1 => addr(7),
      O => mem_reg_0_127_0_0_i_1_n_0
    );
mem_reg_0_15_0_0: unisim.vcomponents.RAM32X1S
    generic map(
      INIT => X"00000000"
    )
        port map (
      A0 => addr(0),
      A1 => addr(1),
      A2 => addr(2),
      A3 => addr(3),
      A4 => '0',
      D => d_in(0),
      O => mem_reg_0_15_0_0_n_0,
      WCLK => clk,
      WE => mem_reg_0_15_0_0_i_1_n_0
    );
\mem_reg_0_15_0_0__0\: unisim.vcomponents.RAM32X1S
    generic map(
      INIT => X"00000000"
    )
        port map (
      A0 => addr(0),
      A1 => addr(1),
      A2 => addr(2),
      A3 => addr(3),
      A4 => '0',
      D => d_in(1),
      O => \mem_reg_0_15_0_0__0_n_0\,
      WCLK => clk,
      WE => mem_reg_0_15_0_0_i_1_n_0
    );
\mem_reg_0_15_0_0__1\: unisim.vcomponents.RAM32X1S
    generic map(
      INIT => X"00000000"
    )
        port map (
      A0 => addr(0),
      A1 => addr(1),
      A2 => addr(2),
      A3 => addr(3),
      A4 => '0',
      D => d_in(2),
      O => \mem_reg_0_15_0_0__1_n_0\,
      WCLK => clk,
      WE => mem_reg_0_15_0_0_i_1_n_0
    );
\mem_reg_0_15_0_0__10\: unisim.vcomponents.RAM32X1S
    generic map(
      INIT => X"00000000"
    )
        port map (
      A0 => addr(0),
      A1 => addr(1),
      A2 => addr(2),
      A3 => addr(3),
      A4 => '0',
      D => d_in(11),
      O => \mem_reg_0_15_0_0__10_n_0\,
      WCLK => clk,
      WE => mem_reg_0_15_0_0_i_1_n_0
    );
\mem_reg_0_15_0_0__11\: unisim.vcomponents.RAM32X1S
    generic map(
      INIT => X"00000000"
    )
        port map (
      A0 => addr(0),
      A1 => addr(1),
      A2 => addr(2),
      A3 => addr(3),
      A4 => '0',
      D => d_in(12),
      O => \mem_reg_0_15_0_0__11_n_0\,
      WCLK => clk,
      WE => mem_reg_0_15_0_0_i_1_n_0
    );
\mem_reg_0_15_0_0__12\: unisim.vcomponents.RAM32X1S
    generic map(
      INIT => X"00000000"
    )
        port map (
      A0 => addr(0),
      A1 => addr(1),
      A2 => addr(2),
      A3 => addr(3),
      A4 => '0',
      D => d_in(13),
      O => \mem_reg_0_15_0_0__12_n_0\,
      WCLK => clk,
      WE => mem_reg_0_15_0_0_i_1_n_0
    );
\mem_reg_0_15_0_0__13\: unisim.vcomponents.RAM32X1S
    generic map(
      INIT => X"00000000"
    )
        port map (
      A0 => addr(0),
      A1 => addr(1),
      A2 => addr(2),
      A3 => addr(3),
      A4 => '0',
      D => d_in(14),
      O => \mem_reg_0_15_0_0__13_n_0\,
      WCLK => clk,
      WE => mem_reg_0_15_0_0_i_1_n_0
    );
\mem_reg_0_15_0_0__14\: unisim.vcomponents.RAM32X1S
    generic map(
      INIT => X"00000000"
    )
        port map (
      A0 => addr(0),
      A1 => addr(1),
      A2 => addr(2),
      A3 => addr(3),
      A4 => '0',
      D => d_in(15),
      O => \mem_reg_0_15_0_0__14_n_0\,
      WCLK => clk,
      WE => mem_reg_0_15_0_0_i_1_n_0
    );
\mem_reg_0_15_0_0__15\: unisim.vcomponents.RAM32X1S
    generic map(
      INIT => X"00000000"
    )
        port map (
      A0 => addr(0),
      A1 => addr(1),
      A2 => addr(2),
      A3 => addr(3),
      A4 => '0',
      D => d_in(16),
      O => \mem_reg_0_15_0_0__15_n_0\,
      WCLK => clk,
      WE => mem_reg_0_15_0_0_i_1_n_0
    );
\mem_reg_0_15_0_0__16\: unisim.vcomponents.RAM32X1S
    generic map(
      INIT => X"00000000"
    )
        port map (
      A0 => addr(0),
      A1 => addr(1),
      A2 => addr(2),
      A3 => addr(3),
      A4 => '0',
      D => d_in(17),
      O => \mem_reg_0_15_0_0__16_n_0\,
      WCLK => clk,
      WE => mem_reg_0_15_0_0_i_1_n_0
    );
\mem_reg_0_15_0_0__17\: unisim.vcomponents.RAM32X1S
    generic map(
      INIT => X"00000000"
    )
        port map (
      A0 => addr(0),
      A1 => addr(1),
      A2 => addr(2),
      A3 => addr(3),
      A4 => '0',
      D => d_in(18),
      O => \mem_reg_0_15_0_0__17_n_0\,
      WCLK => clk,
      WE => mem_reg_0_15_0_0_i_1_n_0
    );
\mem_reg_0_15_0_0__18\: unisim.vcomponents.RAM32X1S
    generic map(
      INIT => X"00000000"
    )
        port map (
      A0 => addr(0),
      A1 => addr(1),
      A2 => addr(2),
      A3 => addr(3),
      A4 => '0',
      D => d_in(19),
      O => \mem_reg_0_15_0_0__18_n_0\,
      WCLK => clk,
      WE => mem_reg_0_15_0_0_i_1_n_0
    );
\mem_reg_0_15_0_0__19\: unisim.vcomponents.RAM32X1S
    generic map(
      INIT => X"00000000"
    )
        port map (
      A0 => addr(0),
      A1 => addr(1),
      A2 => addr(2),
      A3 => addr(3),
      A4 => '0',
      D => d_in(20),
      O => \mem_reg_0_15_0_0__19_n_0\,
      WCLK => clk,
      WE => mem_reg_0_15_0_0_i_1_n_0
    );
\mem_reg_0_15_0_0__2\: unisim.vcomponents.RAM32X1S
    generic map(
      INIT => X"00000000"
    )
        port map (
      A0 => addr(0),
      A1 => addr(1),
      A2 => addr(2),
      A3 => addr(3),
      A4 => '0',
      D => d_in(3),
      O => \mem_reg_0_15_0_0__2_n_0\,
      WCLK => clk,
      WE => mem_reg_0_15_0_0_i_1_n_0
    );
\mem_reg_0_15_0_0__20\: unisim.vcomponents.RAM32X1S
    generic map(
      INIT => X"00000000"
    )
        port map (
      A0 => addr(0),
      A1 => addr(1),
      A2 => addr(2),
      A3 => addr(3),
      A4 => '0',
      D => d_in(21),
      O => \mem_reg_0_15_0_0__20_n_0\,
      WCLK => clk,
      WE => mem_reg_0_15_0_0_i_1_n_0
    );
\mem_reg_0_15_0_0__21\: unisim.vcomponents.RAM32X1S
    generic map(
      INIT => X"00000000"
    )
        port map (
      A0 => addr(0),
      A1 => addr(1),
      A2 => addr(2),
      A3 => addr(3),
      A4 => '0',
      D => d_in(22),
      O => \mem_reg_0_15_0_0__21_n_0\,
      WCLK => clk,
      WE => mem_reg_0_15_0_0_i_1_n_0
    );
\mem_reg_0_15_0_0__22\: unisim.vcomponents.RAM32X1S
    generic map(
      INIT => X"00000000"
    )
        port map (
      A0 => addr(0),
      A1 => addr(1),
      A2 => addr(2),
      A3 => addr(3),
      A4 => '0',
      D => d_in(23),
      O => \mem_reg_0_15_0_0__22_n_0\,
      WCLK => clk,
      WE => mem_reg_0_15_0_0_i_1_n_0
    );
\mem_reg_0_15_0_0__23\: unisim.vcomponents.RAM32X1S
    generic map(
      INIT => X"00000000"
    )
        port map (
      A0 => addr(0),
      A1 => addr(1),
      A2 => addr(2),
      A3 => addr(3),
      A4 => '0',
      D => d_in(24),
      O => \mem_reg_0_15_0_0__23_n_0\,
      WCLK => clk,
      WE => mem_reg_0_15_0_0_i_1_n_0
    );
\mem_reg_0_15_0_0__24\: unisim.vcomponents.RAM32X1S
    generic map(
      INIT => X"00000000"
    )
        port map (
      A0 => addr(0),
      A1 => addr(1),
      A2 => addr(2),
      A3 => addr(3),
      A4 => '0',
      D => d_in(25),
      O => \mem_reg_0_15_0_0__24_n_0\,
      WCLK => clk,
      WE => mem_reg_0_15_0_0_i_1_n_0
    );
\mem_reg_0_15_0_0__25\: unisim.vcomponents.RAM32X1S
    generic map(
      INIT => X"00000000"
    )
        port map (
      A0 => addr(0),
      A1 => addr(1),
      A2 => addr(2),
      A3 => addr(3),
      A4 => '0',
      D => d_in(26),
      O => \mem_reg_0_15_0_0__25_n_0\,
      WCLK => clk,
      WE => mem_reg_0_15_0_0_i_1_n_0
    );
\mem_reg_0_15_0_0__26\: unisim.vcomponents.RAM32X1S
    generic map(
      INIT => X"00000000"
    )
        port map (
      A0 => addr(0),
      A1 => addr(1),
      A2 => addr(2),
      A3 => addr(3),
      A4 => '0',
      D => d_in(27),
      O => \mem_reg_0_15_0_0__26_n_0\,
      WCLK => clk,
      WE => mem_reg_0_15_0_0_i_1_n_0
    );
\mem_reg_0_15_0_0__27\: unisim.vcomponents.RAM32X1S
    generic map(
      INIT => X"00000000"
    )
        port map (
      A0 => addr(0),
      A1 => addr(1),
      A2 => addr(2),
      A3 => addr(3),
      A4 => '0',
      D => d_in(28),
      O => \mem_reg_0_15_0_0__27_n_0\,
      WCLK => clk,
      WE => mem_reg_0_15_0_0_i_1_n_0
    );
\mem_reg_0_15_0_0__28\: unisim.vcomponents.RAM32X1S
    generic map(
      INIT => X"00000000"
    )
        port map (
      A0 => addr(0),
      A1 => addr(1),
      A2 => addr(2),
      A3 => addr(3),
      A4 => '0',
      D => d_in(29),
      O => \mem_reg_0_15_0_0__28_n_0\,
      WCLK => clk,
      WE => mem_reg_0_15_0_0_i_1_n_0
    );
\mem_reg_0_15_0_0__29\: unisim.vcomponents.RAM32X1S
    generic map(
      INIT => X"00000000"
    )
        port map (
      A0 => addr(0),
      A1 => addr(1),
      A2 => addr(2),
      A3 => addr(3),
      A4 => '0',
      D => d_in(30),
      O => \mem_reg_0_15_0_0__29_n_0\,
      WCLK => clk,
      WE => mem_reg_0_15_0_0_i_1_n_0
    );
\mem_reg_0_15_0_0__3\: unisim.vcomponents.RAM32X1S
    generic map(
      INIT => X"00000000"
    )
        port map (
      A0 => addr(0),
      A1 => addr(1),
      A2 => addr(2),
      A3 => addr(3),
      A4 => '0',
      D => d_in(4),
      O => \mem_reg_0_15_0_0__3_n_0\,
      WCLK => clk,
      WE => mem_reg_0_15_0_0_i_1_n_0
    );
\mem_reg_0_15_0_0__30\: unisim.vcomponents.RAM32X1S
    generic map(
      INIT => X"00000000"
    )
        port map (
      A0 => addr(0),
      A1 => addr(1),
      A2 => addr(2),
      A3 => addr(3),
      A4 => '0',
      D => d_in(31),
      O => \mem_reg_0_15_0_0__30_n_0\,
      WCLK => clk,
      WE => mem_reg_0_15_0_0_i_1_n_0
    );
\mem_reg_0_15_0_0__4\: unisim.vcomponents.RAM32X1S
    generic map(
      INIT => X"00000000"
    )
        port map (
      A0 => addr(0),
      A1 => addr(1),
      A2 => addr(2),
      A3 => addr(3),
      A4 => '0',
      D => d_in(5),
      O => \mem_reg_0_15_0_0__4_n_0\,
      WCLK => clk,
      WE => mem_reg_0_15_0_0_i_1_n_0
    );
\mem_reg_0_15_0_0__5\: unisim.vcomponents.RAM32X1S
    generic map(
      INIT => X"00000000"
    )
        port map (
      A0 => addr(0),
      A1 => addr(1),
      A2 => addr(2),
      A3 => addr(3),
      A4 => '0',
      D => d_in(6),
      O => \mem_reg_0_15_0_0__5_n_0\,
      WCLK => clk,
      WE => mem_reg_0_15_0_0_i_1_n_0
    );
\mem_reg_0_15_0_0__6\: unisim.vcomponents.RAM32X1S
    generic map(
      INIT => X"00000000"
    )
        port map (
      A0 => addr(0),
      A1 => addr(1),
      A2 => addr(2),
      A3 => addr(3),
      A4 => '0',
      D => d_in(7),
      O => \mem_reg_0_15_0_0__6_n_0\,
      WCLK => clk,
      WE => mem_reg_0_15_0_0_i_1_n_0
    );
\mem_reg_0_15_0_0__7\: unisim.vcomponents.RAM32X1S
    generic map(
      INIT => X"00000000"
    )
        port map (
      A0 => addr(0),
      A1 => addr(1),
      A2 => addr(2),
      A3 => addr(3),
      A4 => '0',
      D => d_in(8),
      O => \mem_reg_0_15_0_0__7_n_0\,
      WCLK => clk,
      WE => mem_reg_0_15_0_0_i_1_n_0
    );
\mem_reg_0_15_0_0__8\: unisim.vcomponents.RAM32X1S
    generic map(
      INIT => X"00000000"
    )
        port map (
      A0 => addr(0),
      A1 => addr(1),
      A2 => addr(2),
      A3 => addr(3),
      A4 => '0',
      D => d_in(9),
      O => \mem_reg_0_15_0_0__8_n_0\,
      WCLK => clk,
      WE => mem_reg_0_15_0_0_i_1_n_0
    );
\mem_reg_0_15_0_0__9\: unisim.vcomponents.RAM32X1S
    generic map(
      INIT => X"00000000"
    )
        port map (
      A0 => addr(0),
      A1 => addr(1),
      A2 => addr(2),
      A3 => addr(3),
      A4 => '0',
      D => d_in(10),
      O => \mem_reg_0_15_0_0__9_n_0\,
      WCLK => clk,
      WE => mem_reg_0_15_0_0_i_1_n_0
    );
mem_reg_0_15_0_0_i_1: unisim.vcomponents.LUT5
    generic map(
      INIT => X"00020000"
    )
        port map (
      I0 => w_ena,
      I1 => addr(5),
      I2 => addr(4),
      I3 => addr(6),
      I4 => addr(7),
      O => mem_reg_0_15_0_0_i_1_n_0
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
    addr : in STD_LOGIC_VECTOR ( 31 downto 0 );
    d_in : in STD_LOGIC_VECTOR ( 31 downto 0 );
    d_out : out STD_LOGIC_VECTOR ( 31 downto 0 )
  );
  attribute NotValidForBitStream : boolean;
  attribute NotValidForBitStream of decalper_eb_ot_sdeen_pot_pi_dehcac_xnilix : entity is true;
  attribute CHECK_LICENSE_TYPE : string;
  attribute CHECK_LICENSE_TYPE of decalper_eb_ot_sdeen_pot_pi_dehcac_xnilix : entity is "risc32_dmem_0_0,dmem,{}";
  attribute DowngradeIPIdentifiedWarnings : string;
  attribute DowngradeIPIdentifiedWarnings of decalper_eb_ot_sdeen_pot_pi_dehcac_xnilix : entity is "yes";
  attribute IP_DEFINITION_SOURCE : string;
  attribute IP_DEFINITION_SOURCE of decalper_eb_ot_sdeen_pot_pi_dehcac_xnilix : entity is "module_ref";
  attribute X_CORE_INFO : string;
  attribute X_CORE_INFO of decalper_eb_ot_sdeen_pot_pi_dehcac_xnilix : entity is "dmem,Vivado 2022.2";
end decalper_eb_ot_sdeen_pot_pi_dehcac_xnilix;

architecture STRUCTURE of decalper_eb_ot_sdeen_pot_pi_dehcac_xnilix is
  attribute X_INTERFACE_INFO : string;
  attribute X_INTERFACE_INFO of clk : signal is "xilinx.com:signal:clock:1.0 clk CLK";
  attribute X_INTERFACE_PARAMETER : string;
  attribute X_INTERFACE_PARAMETER of clk : signal is "XIL_INTERFACENAME clk, FREQ_HZ 100000000, FREQ_TOLERANCE_HZ 0, PHASE 0.0, CLK_DOMAIN risc32_clk_0, INSERT_VIP 0";
begin
inst: entity work.decalper_eb_ot_sdeen_pot_pi_dehcac_xnilix_dmem
     port map (
      addr(7 downto 0) => addr(7 downto 0),
      clk => clk,
      d_in(31 downto 0) => d_in(31 downto 0),
      d_out(31 downto 0) => d_out(31 downto 0),
      w_ena => w_ena
    );
end STRUCTURE;
