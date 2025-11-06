-- Copyright 1986-2022 Xilinx, Inc. All Rights Reserved.
-- --------------------------------------------------------------------------------
-- Tool Version: Vivado v.2022.2 (win64) Build 3671981 Fri Oct 14 05:00:03 MDT 2022
-- Date        : Mon Oct 28 16:04:54 2024
-- Host        : cop running 64-bit major release  (build 9200)
-- Command     : write_vhdl -force -mode funcsim -rename_top decalper_eb_ot_sdeen_pot_pi_dehcac_xnilix -prefix
--               decalper_eb_ot_sdeen_pot_pi_dehcac_xnilix_ risc32_alu32_0_0_sim_netlist.vhdl
-- Design      : risc32_alu32_0_0
-- Purpose     : This VHDL netlist is a functional simulation representation of the design and should not be modified or
--               synthesized. This netlist cannot be used for SDF annotated simulation.
-- Device      : xc7a100tcsg324-1
-- --------------------------------------------------------------------------------
library IEEE;
use IEEE.STD_LOGIC_1164.ALL;
library UNISIM;
use UNISIM.VCOMPONENTS.ALL;
entity decalper_eb_ot_sdeen_pot_pi_dehcac_xnilix_alu32 is
  port (
    alu_out : out STD_LOGIC_VECTOR ( 31 downto 0 );
    alu_a : in STD_LOGIC_VECTOR ( 31 downto 0 );
    alu_b : in STD_LOGIC_VECTOR ( 31 downto 0 );
    alu_out_31_sp_1 : in STD_LOGIC;
    alu_sel : in STD_LOGIC_VECTOR ( 2 downto 0 );
    \alu_out[31]_0\ : in STD_LOGIC;
    \alu_out[31]_1\ : in STD_LOGIC;
    \alu_out[31]_2\ : in STD_LOGIC;
    alu_out_1_sp_1 : in STD_LOGIC;
    \alu_out[1]_0\ : in STD_LOGIC;
    \alu_out[1]_1\ : in STD_LOGIC;
    alu_out_2_sp_1 : in STD_LOGIC;
    \alu_out[2]_0\ : in STD_LOGIC;
    alu_out_17_sp_1 : in STD_LOGIC;
    \alu_out[17]_0\ : in STD_LOGIC;
    \alu_out[17]_1\ : in STD_LOGIC;
    alu_out_18_sp_1 : in STD_LOGIC;
    \alu_out[18]_0\ : in STD_LOGIC;
    alu_out_15_sp_1 : in STD_LOGIC;
    \alu_out[15]_0\ : in STD_LOGIC;
    \alu_out[15]_1\ : in STD_LOGIC;
    alu_out_16_sp_1 : in STD_LOGIC;
    alu_out_13_sp_1 : in STD_LOGIC;
    \alu_out[13]_0\ : in STD_LOGIC;
    \alu_out[13]_1\ : in STD_LOGIC;
    alu_out_14_sp_1 : in STD_LOGIC;
    alu_out_11_sp_1 : in STD_LOGIC;
    \alu_out[11]_0\ : in STD_LOGIC;
    \alu_out[11]_1\ : in STD_LOGIC;
    alu_out_12_sp_1 : in STD_LOGIC;
    alu_out_4_sp_1 : in STD_LOGIC;
    \alu_out[4]_0\ : in STD_LOGIC;
    \alu_out[4]_1\ : in STD_LOGIC;
    alu_out_5_sp_1 : in STD_LOGIC;
    \alu_out[5]_0\ : in STD_LOGIC;
    alu_out_6_sp_1 : in STD_LOGIC;
    \alu_out[6]_0\ : in STD_LOGIC;
    alu_out_7_sp_1 : in STD_LOGIC;
    \alu_out[7]_0\ : in STD_LOGIC;
    alu_out_8_sp_1 : in STD_LOGIC;
    \alu_out[8]_0\ : in STD_LOGIC;
    alu_out_9_sp_1 : in STD_LOGIC;
    \alu_out[9]_0\ : in STD_LOGIC;
    alu_out_10_sp_1 : in STD_LOGIC;
    alu_out_19_sp_1 : in STD_LOGIC;
    \alu_out[19]_0\ : in STD_LOGIC;
    alu_out_20_sp_1 : in STD_LOGIC;
    \alu_out[20]_0\ : in STD_LOGIC;
    alu_out_21_sp_1 : in STD_LOGIC;
    \alu_out[21]_0\ : in STD_LOGIC;
    alu_out_22_sp_1 : in STD_LOGIC;
    \alu_out[22]_0\ : in STD_LOGIC;
    alu_out_23_sp_1 : in STD_LOGIC;
    \alu_out[23]_0\ : in STD_LOGIC;
    alu_out_24_sp_1 : in STD_LOGIC;
    \alu_out[24]_0\ : in STD_LOGIC;
    alu_out_25_sp_1 : in STD_LOGIC;
    \alu_out[25]_0\ : in STD_LOGIC;
    alu_out_26_sp_1 : in STD_LOGIC;
    \alu_out[26]_0\ : in STD_LOGIC;
    alu_out_27_sp_1 : in STD_LOGIC;
    \alu_out[27]_0\ : in STD_LOGIC;
    alu_out_28_sp_1 : in STD_LOGIC;
    \alu_out[28]_0\ : in STD_LOGIC;
    alu_out_29_sp_1 : in STD_LOGIC;
    \alu_out[29]_0\ : in STD_LOGIC;
    alu_out_30_sp_1 : in STD_LOGIC;
    alu_out_0_sp_1 : in STD_LOGIC;
    alu_out_3_sp_1 : in STD_LOGIC
  );
end decalper_eb_ot_sdeen_pot_pi_dehcac_xnilix_alu32;

architecture STRUCTURE of decalper_eb_ot_sdeen_pot_pi_dehcac_xnilix_alu32 is
  signal \alu_out0__93_carry__0_n_0\ : STD_LOGIC;
  signal \alu_out0__93_carry__0_n_1\ : STD_LOGIC;
  signal \alu_out0__93_carry__0_n_2\ : STD_LOGIC;
  signal \alu_out0__93_carry__0_n_3\ : STD_LOGIC;
  signal \alu_out0__93_carry__1_n_0\ : STD_LOGIC;
  signal \alu_out0__93_carry__1_n_1\ : STD_LOGIC;
  signal \alu_out0__93_carry__1_n_2\ : STD_LOGIC;
  signal \alu_out0__93_carry__1_n_3\ : STD_LOGIC;
  signal \alu_out0__93_carry__2_n_1\ : STD_LOGIC;
  signal \alu_out0__93_carry__2_n_2\ : STD_LOGIC;
  signal \alu_out0__93_carry__2_n_3\ : STD_LOGIC;
  signal \alu_out0__93_carry_i_1__0_n_0\ : STD_LOGIC;
  signal \alu_out0__93_carry_i_1__1_n_0\ : STD_LOGIC;
  signal \alu_out0__93_carry_i_1__2_n_0\ : STD_LOGIC;
  signal \alu_out0__93_carry_i_1_n_0\ : STD_LOGIC;
  signal \alu_out0__93_carry_i_2__0_n_0\ : STD_LOGIC;
  signal \alu_out0__93_carry_i_2__1_n_0\ : STD_LOGIC;
  signal \alu_out0__93_carry_i_2__2_n_0\ : STD_LOGIC;
  signal \alu_out0__93_carry_i_2_n_0\ : STD_LOGIC;
  signal \alu_out0__93_carry_i_3__0_n_0\ : STD_LOGIC;
  signal \alu_out0__93_carry_i_3__1_n_0\ : STD_LOGIC;
  signal \alu_out0__93_carry_i_3__2_n_0\ : STD_LOGIC;
  signal \alu_out0__93_carry_i_3_n_0\ : STD_LOGIC;
  signal \alu_out0__93_carry_i_4__0_n_0\ : STD_LOGIC;
  signal \alu_out0__93_carry_i_4__1_n_0\ : STD_LOGIC;
  signal \alu_out0__93_carry_i_4__2_n_0\ : STD_LOGIC;
  signal \alu_out0__93_carry_i_4_n_0\ : STD_LOGIC;
  signal \alu_out0__93_carry_i_5__0_n_0\ : STD_LOGIC;
  signal \alu_out0__93_carry_i_5__1_n_0\ : STD_LOGIC;
  signal \alu_out0__93_carry_i_5__2_n_0\ : STD_LOGIC;
  signal \alu_out0__93_carry_i_5_n_0\ : STD_LOGIC;
  signal \alu_out0__93_carry_i_6__0_n_0\ : STD_LOGIC;
  signal \alu_out0__93_carry_i_6__1_n_0\ : STD_LOGIC;
  signal \alu_out0__93_carry_i_6__2_n_0\ : STD_LOGIC;
  signal \alu_out0__93_carry_i_6_n_0\ : STD_LOGIC;
  signal \alu_out0__93_carry_i_7__0_n_0\ : STD_LOGIC;
  signal \alu_out0__93_carry_i_7__1_n_0\ : STD_LOGIC;
  signal \alu_out0__93_carry_i_7__2_n_0\ : STD_LOGIC;
  signal \alu_out0__93_carry_i_7_n_0\ : STD_LOGIC;
  signal \alu_out0__93_carry_i_8__0_n_0\ : STD_LOGIC;
  signal \alu_out0__93_carry_i_8__1_n_0\ : STD_LOGIC;
  signal \alu_out0__93_carry_i_8__2_n_0\ : STD_LOGIC;
  signal \alu_out0__93_carry_i_8_n_0\ : STD_LOGIC;
  signal \alu_out0__93_carry_n_0\ : STD_LOGIC;
  signal \alu_out0__93_carry_n_1\ : STD_LOGIC;
  signal \alu_out0__93_carry_n_2\ : STD_LOGIC;
  signal \alu_out0__93_carry_n_3\ : STD_LOGIC;
  signal \alu_out0_carry__0_i_1_n_0\ : STD_LOGIC;
  signal \alu_out0_carry__0_i_2_n_0\ : STD_LOGIC;
  signal \alu_out0_carry__0_i_3_n_0\ : STD_LOGIC;
  signal \alu_out0_carry__0_i_4_n_0\ : STD_LOGIC;
  signal \alu_out0_carry__0_n_0\ : STD_LOGIC;
  signal \alu_out0_carry__0_n_1\ : STD_LOGIC;
  signal \alu_out0_carry__0_n_2\ : STD_LOGIC;
  signal \alu_out0_carry__0_n_3\ : STD_LOGIC;
  signal \alu_out0_carry__1_i_1_n_0\ : STD_LOGIC;
  signal \alu_out0_carry__1_i_2_n_0\ : STD_LOGIC;
  signal \alu_out0_carry__1_i_3_n_0\ : STD_LOGIC;
  signal \alu_out0_carry__1_i_4_n_0\ : STD_LOGIC;
  signal \alu_out0_carry__1_n_0\ : STD_LOGIC;
  signal \alu_out0_carry__1_n_1\ : STD_LOGIC;
  signal \alu_out0_carry__1_n_2\ : STD_LOGIC;
  signal \alu_out0_carry__1_n_3\ : STD_LOGIC;
  signal \alu_out0_carry__2_i_1_n_0\ : STD_LOGIC;
  signal \alu_out0_carry__2_i_2_n_0\ : STD_LOGIC;
  signal \alu_out0_carry__2_i_3_n_0\ : STD_LOGIC;
  signal \alu_out0_carry__2_i_4_n_0\ : STD_LOGIC;
  signal \alu_out0_carry__2_n_0\ : STD_LOGIC;
  signal \alu_out0_carry__2_n_1\ : STD_LOGIC;
  signal \alu_out0_carry__2_n_2\ : STD_LOGIC;
  signal \alu_out0_carry__2_n_3\ : STD_LOGIC;
  signal \alu_out0_carry__3_i_1_n_0\ : STD_LOGIC;
  signal \alu_out0_carry__3_i_2_n_0\ : STD_LOGIC;
  signal \alu_out0_carry__3_i_3_n_0\ : STD_LOGIC;
  signal \alu_out0_carry__3_i_4_n_0\ : STD_LOGIC;
  signal \alu_out0_carry__3_n_0\ : STD_LOGIC;
  signal \alu_out0_carry__3_n_1\ : STD_LOGIC;
  signal \alu_out0_carry__3_n_2\ : STD_LOGIC;
  signal \alu_out0_carry__3_n_3\ : STD_LOGIC;
  signal \alu_out0_carry__4_i_1_n_0\ : STD_LOGIC;
  signal \alu_out0_carry__4_i_2_n_0\ : STD_LOGIC;
  signal \alu_out0_carry__4_i_3_n_0\ : STD_LOGIC;
  signal \alu_out0_carry__4_i_4_n_0\ : STD_LOGIC;
  signal \alu_out0_carry__4_n_0\ : STD_LOGIC;
  signal \alu_out0_carry__4_n_1\ : STD_LOGIC;
  signal \alu_out0_carry__4_n_2\ : STD_LOGIC;
  signal \alu_out0_carry__4_n_3\ : STD_LOGIC;
  signal \alu_out0_carry__5_i_1_n_0\ : STD_LOGIC;
  signal \alu_out0_carry__5_i_2_n_0\ : STD_LOGIC;
  signal \alu_out0_carry__5_i_3_n_0\ : STD_LOGIC;
  signal \alu_out0_carry__5_i_4_n_0\ : STD_LOGIC;
  signal \alu_out0_carry__5_n_0\ : STD_LOGIC;
  signal \alu_out0_carry__5_n_1\ : STD_LOGIC;
  signal \alu_out0_carry__5_n_2\ : STD_LOGIC;
  signal \alu_out0_carry__5_n_3\ : STD_LOGIC;
  signal \alu_out0_carry__6_i_1_n_0\ : STD_LOGIC;
  signal \alu_out0_carry__6_i_2_n_0\ : STD_LOGIC;
  signal \alu_out0_carry__6_i_3_n_0\ : STD_LOGIC;
  signal \alu_out0_carry__6_i_4_n_0\ : STD_LOGIC;
  signal \alu_out0_carry__6_n_1\ : STD_LOGIC;
  signal \alu_out0_carry__6_n_2\ : STD_LOGIC;
  signal \alu_out0_carry__6_n_3\ : STD_LOGIC;
  signal alu_out0_carry_i_1_n_0 : STD_LOGIC;
  signal alu_out0_carry_i_2_n_0 : STD_LOGIC;
  signal alu_out0_carry_i_3_n_0 : STD_LOGIC;
  signal alu_out0_carry_i_4_n_0 : STD_LOGIC;
  signal alu_out0_carry_n_0 : STD_LOGIC;
  signal alu_out0_carry_n_1 : STD_LOGIC;
  signal alu_out0_carry_n_2 : STD_LOGIC;
  signal alu_out0_carry_n_3 : STD_LOGIC;
  signal \alu_out[0]_INST_0_i_2_n_0\ : STD_LOGIC;
  signal \alu_out[0]_INST_0_i_3_n_0\ : STD_LOGIC;
  signal \alu_out[10]_INST_0_i_3_n_0\ : STD_LOGIC;
  signal \alu_out[11]_INST_0_i_3_n_0\ : STD_LOGIC;
  signal \alu_out[12]_INST_0_i_3_n_0\ : STD_LOGIC;
  signal \alu_out[13]_INST_0_i_3_n_0\ : STD_LOGIC;
  signal \alu_out[14]_INST_0_i_3_n_0\ : STD_LOGIC;
  signal \alu_out[15]_INST_0_i_3_n_0\ : STD_LOGIC;
  signal \alu_out[16]_INST_0_i_3_n_0\ : STD_LOGIC;
  signal \alu_out[17]_INST_0_i_3_n_0\ : STD_LOGIC;
  signal \alu_out[18]_INST_0_i_3_n_0\ : STD_LOGIC;
  signal \alu_out[19]_INST_0_i_3_n_0\ : STD_LOGIC;
  signal \alu_out[1]_INST_0_i_3_n_0\ : STD_LOGIC;
  signal \alu_out[20]_INST_0_i_3_n_0\ : STD_LOGIC;
  signal \alu_out[21]_INST_0_i_3_n_0\ : STD_LOGIC;
  signal \alu_out[22]_INST_0_i_3_n_0\ : STD_LOGIC;
  signal \alu_out[23]_INST_0_i_3_n_0\ : STD_LOGIC;
  signal \alu_out[24]_INST_0_i_3_n_0\ : STD_LOGIC;
  signal \alu_out[25]_INST_0_i_3_n_0\ : STD_LOGIC;
  signal \alu_out[26]_INST_0_i_3_n_0\ : STD_LOGIC;
  signal \alu_out[27]_INST_0_i_3_n_0\ : STD_LOGIC;
  signal \alu_out[28]_INST_0_i_3_n_0\ : STD_LOGIC;
  signal \alu_out[29]_INST_0_i_3_n_0\ : STD_LOGIC;
  signal \alu_out[2]_INST_0_i_3_n_0\ : STD_LOGIC;
  signal \alu_out[30]_INST_0_i_3_n_0\ : STD_LOGIC;
  signal \alu_out[31]_INST_0_i_5_n_0\ : STD_LOGIC;
  signal \alu_out[3]_INST_0_i_3_n_0\ : STD_LOGIC;
  signal \alu_out[4]_INST_0_i_3_n_0\ : STD_LOGIC;
  signal \alu_out[5]_INST_0_i_3_n_0\ : STD_LOGIC;
  signal \alu_out[6]_INST_0_i_3_n_0\ : STD_LOGIC;
  signal \alu_out[7]_INST_0_i_3_n_0\ : STD_LOGIC;
  signal \alu_out[8]_INST_0_i_3_n_0\ : STD_LOGIC;
  signal \alu_out[9]_INST_0_i_3_n_0\ : STD_LOGIC;
  signal alu_out_0_sn_1 : STD_LOGIC;
  signal alu_out_10_sn_1 : STD_LOGIC;
  signal alu_out_11_sn_1 : STD_LOGIC;
  signal alu_out_12_sn_1 : STD_LOGIC;
  signal alu_out_13_sn_1 : STD_LOGIC;
  signal alu_out_14_sn_1 : STD_LOGIC;
  signal alu_out_15_sn_1 : STD_LOGIC;
  signal alu_out_16_sn_1 : STD_LOGIC;
  signal alu_out_17_sn_1 : STD_LOGIC;
  signal alu_out_18_sn_1 : STD_LOGIC;
  signal alu_out_19_sn_1 : STD_LOGIC;
  signal alu_out_1_sn_1 : STD_LOGIC;
  signal alu_out_20_sn_1 : STD_LOGIC;
  signal alu_out_21_sn_1 : STD_LOGIC;
  signal alu_out_22_sn_1 : STD_LOGIC;
  signal alu_out_23_sn_1 : STD_LOGIC;
  signal alu_out_24_sn_1 : STD_LOGIC;
  signal alu_out_25_sn_1 : STD_LOGIC;
  signal alu_out_26_sn_1 : STD_LOGIC;
  signal alu_out_27_sn_1 : STD_LOGIC;
  signal alu_out_28_sn_1 : STD_LOGIC;
  signal alu_out_29_sn_1 : STD_LOGIC;
  signal alu_out_2_sn_1 : STD_LOGIC;
  signal alu_out_30_sn_1 : STD_LOGIC;
  signal alu_out_31_sn_1 : STD_LOGIC;
  signal alu_out_3_sn_1 : STD_LOGIC;
  signal alu_out_4_sn_1 : STD_LOGIC;
  signal alu_out_5_sn_1 : STD_LOGIC;
  signal alu_out_6_sn_1 : STD_LOGIC;
  signal alu_out_7_sn_1 : STD_LOGIC;
  signal alu_out_8_sn_1 : STD_LOGIC;
  signal alu_out_9_sn_1 : STD_LOGIC;
  signal data0 : STD_LOGIC_VECTOR ( 31 downto 0 );
  signal data4 : STD_LOGIC;
  signal \NLW_alu_out0__93_carry_O_UNCONNECTED\ : STD_LOGIC_VECTOR ( 3 downto 0 );
  signal \NLW_alu_out0__93_carry__0_O_UNCONNECTED\ : STD_LOGIC_VECTOR ( 3 downto 0 );
  signal \NLW_alu_out0__93_carry__1_O_UNCONNECTED\ : STD_LOGIC_VECTOR ( 3 downto 0 );
  signal \NLW_alu_out0__93_carry__2_O_UNCONNECTED\ : STD_LOGIC_VECTOR ( 3 downto 0 );
  signal \NLW_alu_out0_carry__6_CO_UNCONNECTED\ : STD_LOGIC_VECTOR ( 3 to 3 );
  attribute COMPARATOR_THRESHOLD : integer;
  attribute COMPARATOR_THRESHOLD of \alu_out0__93_carry\ : label is 11;
  attribute COMPARATOR_THRESHOLD of \alu_out0__93_carry__0\ : label is 11;
  attribute COMPARATOR_THRESHOLD of \alu_out0__93_carry__1\ : label is 11;
  attribute COMPARATOR_THRESHOLD of \alu_out0__93_carry__2\ : label is 11;
  attribute ADDER_THRESHOLD : integer;
  attribute ADDER_THRESHOLD of alu_out0_carry : label is 35;
  attribute ADDER_THRESHOLD of \alu_out0_carry__0\ : label is 35;
  attribute ADDER_THRESHOLD of \alu_out0_carry__1\ : label is 35;
  attribute ADDER_THRESHOLD of \alu_out0_carry__2\ : label is 35;
  attribute ADDER_THRESHOLD of \alu_out0_carry__3\ : label is 35;
  attribute ADDER_THRESHOLD of \alu_out0_carry__4\ : label is 35;
  attribute ADDER_THRESHOLD of \alu_out0_carry__5\ : label is 35;
  attribute ADDER_THRESHOLD of \alu_out0_carry__6\ : label is 35;
begin
  alu_out_0_sn_1 <= alu_out_0_sp_1;
  alu_out_10_sn_1 <= alu_out_10_sp_1;
  alu_out_11_sn_1 <= alu_out_11_sp_1;
  alu_out_12_sn_1 <= alu_out_12_sp_1;
  alu_out_13_sn_1 <= alu_out_13_sp_1;
  alu_out_14_sn_1 <= alu_out_14_sp_1;
  alu_out_15_sn_1 <= alu_out_15_sp_1;
  alu_out_16_sn_1 <= alu_out_16_sp_1;
  alu_out_17_sn_1 <= alu_out_17_sp_1;
  alu_out_18_sn_1 <= alu_out_18_sp_1;
  alu_out_19_sn_1 <= alu_out_19_sp_1;
  alu_out_1_sn_1 <= alu_out_1_sp_1;
  alu_out_20_sn_1 <= alu_out_20_sp_1;
  alu_out_21_sn_1 <= alu_out_21_sp_1;
  alu_out_22_sn_1 <= alu_out_22_sp_1;
  alu_out_23_sn_1 <= alu_out_23_sp_1;
  alu_out_24_sn_1 <= alu_out_24_sp_1;
  alu_out_25_sn_1 <= alu_out_25_sp_1;
  alu_out_26_sn_1 <= alu_out_26_sp_1;
  alu_out_27_sn_1 <= alu_out_27_sp_1;
  alu_out_28_sn_1 <= alu_out_28_sp_1;
  alu_out_29_sn_1 <= alu_out_29_sp_1;
  alu_out_2_sn_1 <= alu_out_2_sp_1;
  alu_out_30_sn_1 <= alu_out_30_sp_1;
  alu_out_31_sn_1 <= alu_out_31_sp_1;
  alu_out_3_sn_1 <= alu_out_3_sp_1;
  alu_out_4_sn_1 <= alu_out_4_sp_1;
  alu_out_5_sn_1 <= alu_out_5_sp_1;
  alu_out_6_sn_1 <= alu_out_6_sp_1;
  alu_out_7_sn_1 <= alu_out_7_sp_1;
  alu_out_8_sn_1 <= alu_out_8_sp_1;
  alu_out_9_sn_1 <= alu_out_9_sp_1;
\alu_out0__93_carry\: unisim.vcomponents.CARRY4
     port map (
      CI => '0',
      CO(3) => \alu_out0__93_carry_n_0\,
      CO(2) => \alu_out0__93_carry_n_1\,
      CO(1) => \alu_out0__93_carry_n_2\,
      CO(0) => \alu_out0__93_carry_n_3\,
      CYINIT => '0',
      DI(3) => \alu_out0__93_carry_i_1_n_0\,
      DI(2) => \alu_out0__93_carry_i_2_n_0\,
      DI(1) => \alu_out0__93_carry_i_3_n_0\,
      DI(0) => \alu_out0__93_carry_i_4_n_0\,
      O(3 downto 0) => \NLW_alu_out0__93_carry_O_UNCONNECTED\(3 downto 0),
      S(3) => \alu_out0__93_carry_i_5_n_0\,
      S(2) => \alu_out0__93_carry_i_6_n_0\,
      S(1) => \alu_out0__93_carry_i_7_n_0\,
      S(0) => \alu_out0__93_carry_i_8_n_0\
    );
\alu_out0__93_carry__0\: unisim.vcomponents.CARRY4
     port map (
      CI => \alu_out0__93_carry_n_0\,
      CO(3) => \alu_out0__93_carry__0_n_0\,
      CO(2) => \alu_out0__93_carry__0_n_1\,
      CO(1) => \alu_out0__93_carry__0_n_2\,
      CO(0) => \alu_out0__93_carry__0_n_3\,
      CYINIT => '0',
      DI(3) => \alu_out0__93_carry_i_1__0_n_0\,
      DI(2) => \alu_out0__93_carry_i_2__0_n_0\,
      DI(1) => \alu_out0__93_carry_i_3__0_n_0\,
      DI(0) => \alu_out0__93_carry_i_4__0_n_0\,
      O(3 downto 0) => \NLW_alu_out0__93_carry__0_O_UNCONNECTED\(3 downto 0),
      S(3) => \alu_out0__93_carry_i_5__0_n_0\,
      S(2) => \alu_out0__93_carry_i_6__0_n_0\,
      S(1) => \alu_out0__93_carry_i_7__0_n_0\,
      S(0) => \alu_out0__93_carry_i_8__0_n_0\
    );
\alu_out0__93_carry__1\: unisim.vcomponents.CARRY4
     port map (
      CI => \alu_out0__93_carry__0_n_0\,
      CO(3) => \alu_out0__93_carry__1_n_0\,
      CO(2) => \alu_out0__93_carry__1_n_1\,
      CO(1) => \alu_out0__93_carry__1_n_2\,
      CO(0) => \alu_out0__93_carry__1_n_3\,
      CYINIT => '0',
      DI(3) => \alu_out0__93_carry_i_1__1_n_0\,
      DI(2) => \alu_out0__93_carry_i_2__1_n_0\,
      DI(1) => \alu_out0__93_carry_i_3__1_n_0\,
      DI(0) => \alu_out0__93_carry_i_4__1_n_0\,
      O(3 downto 0) => \NLW_alu_out0__93_carry__1_O_UNCONNECTED\(3 downto 0),
      S(3) => \alu_out0__93_carry_i_5__1_n_0\,
      S(2) => \alu_out0__93_carry_i_6__1_n_0\,
      S(1) => \alu_out0__93_carry_i_7__1_n_0\,
      S(0) => \alu_out0__93_carry_i_8__1_n_0\
    );
\alu_out0__93_carry__2\: unisim.vcomponents.CARRY4
     port map (
      CI => \alu_out0__93_carry__1_n_0\,
      CO(3) => data4,
      CO(2) => \alu_out0__93_carry__2_n_1\,
      CO(1) => \alu_out0__93_carry__2_n_2\,
      CO(0) => \alu_out0__93_carry__2_n_3\,
      CYINIT => '0',
      DI(3) => \alu_out0__93_carry_i_1__2_n_0\,
      DI(2) => \alu_out0__93_carry_i_2__2_n_0\,
      DI(1) => \alu_out0__93_carry_i_3__2_n_0\,
      DI(0) => \alu_out0__93_carry_i_4__2_n_0\,
      O(3 downto 0) => \NLW_alu_out0__93_carry__2_O_UNCONNECTED\(3 downto 0),
      S(3) => \alu_out0__93_carry_i_5__2_n_0\,
      S(2) => \alu_out0__93_carry_i_6__2_n_0\,
      S(1) => \alu_out0__93_carry_i_7__2_n_0\,
      S(0) => \alu_out0__93_carry_i_8__2_n_0\
    );
\alu_out0__93_carry_i_1\: unisim.vcomponents.LUT4
    generic map(
      INIT => X"4F04"
    )
        port map (
      I0 => alu_a(6),
      I1 => alu_b(6),
      I2 => alu_a(7),
      I3 => alu_b(7),
      O => \alu_out0__93_carry_i_1_n_0\
    );
\alu_out0__93_carry_i_1__0\: unisim.vcomponents.LUT4
    generic map(
      INIT => X"4F04"
    )
        port map (
      I0 => alu_a(14),
      I1 => alu_b(14),
      I2 => alu_a(15),
      I3 => alu_b(15),
      O => \alu_out0__93_carry_i_1__0_n_0\
    );
\alu_out0__93_carry_i_1__1\: unisim.vcomponents.LUT4
    generic map(
      INIT => X"4F04"
    )
        port map (
      I0 => alu_a(22),
      I1 => alu_b(22),
      I2 => alu_a(23),
      I3 => alu_b(23),
      O => \alu_out0__93_carry_i_1__1_n_0\
    );
\alu_out0__93_carry_i_1__2\: unisim.vcomponents.LUT4
    generic map(
      INIT => X"44D4"
    )
        port map (
      I0 => alu_b(31),
      I1 => alu_a(31),
      I2 => alu_b(30),
      I3 => alu_a(30),
      O => \alu_out0__93_carry_i_1__2_n_0\
    );
\alu_out0__93_carry_i_2\: unisim.vcomponents.LUT4
    generic map(
      INIT => X"4F04"
    )
        port map (
      I0 => alu_a(4),
      I1 => alu_b(4),
      I2 => alu_a(5),
      I3 => alu_b(5),
      O => \alu_out0__93_carry_i_2_n_0\
    );
\alu_out0__93_carry_i_2__0\: unisim.vcomponents.LUT4
    generic map(
      INIT => X"4F04"
    )
        port map (
      I0 => alu_a(12),
      I1 => alu_b(12),
      I2 => alu_a(13),
      I3 => alu_b(13),
      O => \alu_out0__93_carry_i_2__0_n_0\
    );
\alu_out0__93_carry_i_2__1\: unisim.vcomponents.LUT4
    generic map(
      INIT => X"4F04"
    )
        port map (
      I0 => alu_a(20),
      I1 => alu_b(20),
      I2 => alu_a(21),
      I3 => alu_b(21),
      O => \alu_out0__93_carry_i_2__1_n_0\
    );
\alu_out0__93_carry_i_2__2\: unisim.vcomponents.LUT4
    generic map(
      INIT => X"4F04"
    )
        port map (
      I0 => alu_a(28),
      I1 => alu_b(28),
      I2 => alu_a(29),
      I3 => alu_b(29),
      O => \alu_out0__93_carry_i_2__2_n_0\
    );
\alu_out0__93_carry_i_3\: unisim.vcomponents.LUT4
    generic map(
      INIT => X"4F04"
    )
        port map (
      I0 => alu_a(2),
      I1 => alu_b(2),
      I2 => alu_a(3),
      I3 => alu_b(3),
      O => \alu_out0__93_carry_i_3_n_0\
    );
\alu_out0__93_carry_i_3__0\: unisim.vcomponents.LUT4
    generic map(
      INIT => X"4F04"
    )
        port map (
      I0 => alu_a(10),
      I1 => alu_b(10),
      I2 => alu_a(11),
      I3 => alu_b(11),
      O => \alu_out0__93_carry_i_3__0_n_0\
    );
\alu_out0__93_carry_i_3__1\: unisim.vcomponents.LUT4
    generic map(
      INIT => X"4F04"
    )
        port map (
      I0 => alu_a(18),
      I1 => alu_b(18),
      I2 => alu_a(19),
      I3 => alu_b(19),
      O => \alu_out0__93_carry_i_3__1_n_0\
    );
\alu_out0__93_carry_i_3__2\: unisim.vcomponents.LUT4
    generic map(
      INIT => X"4F04"
    )
        port map (
      I0 => alu_a(26),
      I1 => alu_b(26),
      I2 => alu_a(27),
      I3 => alu_b(27),
      O => \alu_out0__93_carry_i_3__2_n_0\
    );
\alu_out0__93_carry_i_4\: unisim.vcomponents.LUT4
    generic map(
      INIT => X"44D4"
    )
        port map (
      I0 => alu_a(1),
      I1 => alu_b(1),
      I2 => alu_b(0),
      I3 => alu_a(0),
      O => \alu_out0__93_carry_i_4_n_0\
    );
\alu_out0__93_carry_i_4__0\: unisim.vcomponents.LUT4
    generic map(
      INIT => X"4F04"
    )
        port map (
      I0 => alu_a(8),
      I1 => alu_b(8),
      I2 => alu_a(9),
      I3 => alu_b(9),
      O => \alu_out0__93_carry_i_4__0_n_0\
    );
\alu_out0__93_carry_i_4__1\: unisim.vcomponents.LUT4
    generic map(
      INIT => X"4F04"
    )
        port map (
      I0 => alu_a(16),
      I1 => alu_b(16),
      I2 => alu_a(17),
      I3 => alu_b(17),
      O => \alu_out0__93_carry_i_4__1_n_0\
    );
\alu_out0__93_carry_i_4__2\: unisim.vcomponents.LUT4
    generic map(
      INIT => X"4F04"
    )
        port map (
      I0 => alu_a(24),
      I1 => alu_b(24),
      I2 => alu_a(25),
      I3 => alu_b(25),
      O => \alu_out0__93_carry_i_4__2_n_0\
    );
\alu_out0__93_carry_i_5\: unisim.vcomponents.LUT4
    generic map(
      INIT => X"9009"
    )
        port map (
      I0 => alu_b(7),
      I1 => alu_a(7),
      I2 => alu_b(6),
      I3 => alu_a(6),
      O => \alu_out0__93_carry_i_5_n_0\
    );
\alu_out0__93_carry_i_5__0\: unisim.vcomponents.LUT4
    generic map(
      INIT => X"9009"
    )
        port map (
      I0 => alu_b(15),
      I1 => alu_a(15),
      I2 => alu_b(14),
      I3 => alu_a(14),
      O => \alu_out0__93_carry_i_5__0_n_0\
    );
\alu_out0__93_carry_i_5__1\: unisim.vcomponents.LUT4
    generic map(
      INIT => X"9009"
    )
        port map (
      I0 => alu_b(23),
      I1 => alu_a(23),
      I2 => alu_b(22),
      I3 => alu_a(22),
      O => \alu_out0__93_carry_i_5__1_n_0\
    );
\alu_out0__93_carry_i_5__2\: unisim.vcomponents.LUT4
    generic map(
      INIT => X"9009"
    )
        port map (
      I0 => alu_b(30),
      I1 => alu_a(30),
      I2 => alu_a(31),
      I3 => alu_b(31),
      O => \alu_out0__93_carry_i_5__2_n_0\
    );
\alu_out0__93_carry_i_6\: unisim.vcomponents.LUT4
    generic map(
      INIT => X"9009"
    )
        port map (
      I0 => alu_b(5),
      I1 => alu_a(5),
      I2 => alu_b(4),
      I3 => alu_a(4),
      O => \alu_out0__93_carry_i_6_n_0\
    );
\alu_out0__93_carry_i_6__0\: unisim.vcomponents.LUT4
    generic map(
      INIT => X"9009"
    )
        port map (
      I0 => alu_b(13),
      I1 => alu_a(13),
      I2 => alu_b(12),
      I3 => alu_a(12),
      O => \alu_out0__93_carry_i_6__0_n_0\
    );
\alu_out0__93_carry_i_6__1\: unisim.vcomponents.LUT4
    generic map(
      INIT => X"9009"
    )
        port map (
      I0 => alu_b(21),
      I1 => alu_a(21),
      I2 => alu_b(20),
      I3 => alu_a(20),
      O => \alu_out0__93_carry_i_6__1_n_0\
    );
\alu_out0__93_carry_i_6__2\: unisim.vcomponents.LUT4
    generic map(
      INIT => X"9009"
    )
        port map (
      I0 => alu_b(29),
      I1 => alu_a(29),
      I2 => alu_b(28),
      I3 => alu_a(28),
      O => \alu_out0__93_carry_i_6__2_n_0\
    );
\alu_out0__93_carry_i_7\: unisim.vcomponents.LUT4
    generic map(
      INIT => X"9009"
    )
        port map (
      I0 => alu_b(3),
      I1 => alu_a(3),
      I2 => alu_b(2),
      I3 => alu_a(2),
      O => \alu_out0__93_carry_i_7_n_0\
    );
\alu_out0__93_carry_i_7__0\: unisim.vcomponents.LUT4
    generic map(
      INIT => X"9009"
    )
        port map (
      I0 => alu_b(11),
      I1 => alu_a(11),
      I2 => alu_b(10),
      I3 => alu_a(10),
      O => \alu_out0__93_carry_i_7__0_n_0\
    );
\alu_out0__93_carry_i_7__1\: unisim.vcomponents.LUT4
    generic map(
      INIT => X"9009"
    )
        port map (
      I0 => alu_b(19),
      I1 => alu_a(19),
      I2 => alu_b(18),
      I3 => alu_a(18),
      O => \alu_out0__93_carry_i_7__1_n_0\
    );
\alu_out0__93_carry_i_7__2\: unisim.vcomponents.LUT4
    generic map(
      INIT => X"9009"
    )
        port map (
      I0 => alu_b(27),
      I1 => alu_a(27),
      I2 => alu_b(26),
      I3 => alu_a(26),
      O => \alu_out0__93_carry_i_7__2_n_0\
    );
\alu_out0__93_carry_i_8\: unisim.vcomponents.LUT4
    generic map(
      INIT => X"9009"
    )
        port map (
      I0 => alu_b(1),
      I1 => alu_a(1),
      I2 => alu_b(0),
      I3 => alu_a(0),
      O => \alu_out0__93_carry_i_8_n_0\
    );
\alu_out0__93_carry_i_8__0\: unisim.vcomponents.LUT4
    generic map(
      INIT => X"9009"
    )
        port map (
      I0 => alu_b(9),
      I1 => alu_a(9),
      I2 => alu_b(8),
      I3 => alu_a(8),
      O => \alu_out0__93_carry_i_8__0_n_0\
    );
\alu_out0__93_carry_i_8__1\: unisim.vcomponents.LUT4
    generic map(
      INIT => X"9009"
    )
        port map (
      I0 => alu_b(17),
      I1 => alu_a(17),
      I2 => alu_b(16),
      I3 => alu_a(16),
      O => \alu_out0__93_carry_i_8__1_n_0\
    );
\alu_out0__93_carry_i_8__2\: unisim.vcomponents.LUT4
    generic map(
      INIT => X"9009"
    )
        port map (
      I0 => alu_b(25),
      I1 => alu_a(25),
      I2 => alu_b(24),
      I3 => alu_a(24),
      O => \alu_out0__93_carry_i_8__2_n_0\
    );
alu_out0_carry: unisim.vcomponents.CARRY4
     port map (
      CI => '0',
      CO(3) => alu_out0_carry_n_0,
      CO(2) => alu_out0_carry_n_1,
      CO(1) => alu_out0_carry_n_2,
      CO(0) => alu_out0_carry_n_3,
      CYINIT => '0',
      DI(3 downto 0) => alu_a(3 downto 0),
      O(3 downto 0) => data0(3 downto 0),
      S(3) => alu_out0_carry_i_1_n_0,
      S(2) => alu_out0_carry_i_2_n_0,
      S(1) => alu_out0_carry_i_3_n_0,
      S(0) => alu_out0_carry_i_4_n_0
    );
\alu_out0_carry__0\: unisim.vcomponents.CARRY4
     port map (
      CI => alu_out0_carry_n_0,
      CO(3) => \alu_out0_carry__0_n_0\,
      CO(2) => \alu_out0_carry__0_n_1\,
      CO(1) => \alu_out0_carry__0_n_2\,
      CO(0) => \alu_out0_carry__0_n_3\,
      CYINIT => '0',
      DI(3 downto 0) => alu_a(7 downto 4),
      O(3 downto 0) => data0(7 downto 4),
      S(3) => \alu_out0_carry__0_i_1_n_0\,
      S(2) => \alu_out0_carry__0_i_2_n_0\,
      S(1) => \alu_out0_carry__0_i_3_n_0\,
      S(0) => \alu_out0_carry__0_i_4_n_0\
    );
\alu_out0_carry__0_i_1\: unisim.vcomponents.LUT2
    generic map(
      INIT => X"6"
    )
        port map (
      I0 => alu_a(7),
      I1 => alu_b(7),
      O => \alu_out0_carry__0_i_1_n_0\
    );
\alu_out0_carry__0_i_2\: unisim.vcomponents.LUT2
    generic map(
      INIT => X"6"
    )
        port map (
      I0 => alu_a(6),
      I1 => alu_b(6),
      O => \alu_out0_carry__0_i_2_n_0\
    );
\alu_out0_carry__0_i_3\: unisim.vcomponents.LUT2
    generic map(
      INIT => X"6"
    )
        port map (
      I0 => alu_a(5),
      I1 => alu_b(5),
      O => \alu_out0_carry__0_i_3_n_0\
    );
\alu_out0_carry__0_i_4\: unisim.vcomponents.LUT2
    generic map(
      INIT => X"6"
    )
        port map (
      I0 => alu_a(4),
      I1 => alu_b(4),
      O => \alu_out0_carry__0_i_4_n_0\
    );
\alu_out0_carry__1\: unisim.vcomponents.CARRY4
     port map (
      CI => \alu_out0_carry__0_n_0\,
      CO(3) => \alu_out0_carry__1_n_0\,
      CO(2) => \alu_out0_carry__1_n_1\,
      CO(1) => \alu_out0_carry__1_n_2\,
      CO(0) => \alu_out0_carry__1_n_3\,
      CYINIT => '0',
      DI(3 downto 0) => alu_a(11 downto 8),
      O(3 downto 0) => data0(11 downto 8),
      S(3) => \alu_out0_carry__1_i_1_n_0\,
      S(2) => \alu_out0_carry__1_i_2_n_0\,
      S(1) => \alu_out0_carry__1_i_3_n_0\,
      S(0) => \alu_out0_carry__1_i_4_n_0\
    );
\alu_out0_carry__1_i_1\: unisim.vcomponents.LUT2
    generic map(
      INIT => X"6"
    )
        port map (
      I0 => alu_a(11),
      I1 => alu_b(11),
      O => \alu_out0_carry__1_i_1_n_0\
    );
\alu_out0_carry__1_i_2\: unisim.vcomponents.LUT2
    generic map(
      INIT => X"6"
    )
        port map (
      I0 => alu_a(10),
      I1 => alu_b(10),
      O => \alu_out0_carry__1_i_2_n_0\
    );
\alu_out0_carry__1_i_3\: unisim.vcomponents.LUT2
    generic map(
      INIT => X"6"
    )
        port map (
      I0 => alu_a(9),
      I1 => alu_b(9),
      O => \alu_out0_carry__1_i_3_n_0\
    );
\alu_out0_carry__1_i_4\: unisim.vcomponents.LUT2
    generic map(
      INIT => X"6"
    )
        port map (
      I0 => alu_a(8),
      I1 => alu_b(8),
      O => \alu_out0_carry__1_i_4_n_0\
    );
\alu_out0_carry__2\: unisim.vcomponents.CARRY4
     port map (
      CI => \alu_out0_carry__1_n_0\,
      CO(3) => \alu_out0_carry__2_n_0\,
      CO(2) => \alu_out0_carry__2_n_1\,
      CO(1) => \alu_out0_carry__2_n_2\,
      CO(0) => \alu_out0_carry__2_n_3\,
      CYINIT => '0',
      DI(3 downto 0) => alu_a(15 downto 12),
      O(3 downto 0) => data0(15 downto 12),
      S(3) => \alu_out0_carry__2_i_1_n_0\,
      S(2) => \alu_out0_carry__2_i_2_n_0\,
      S(1) => \alu_out0_carry__2_i_3_n_0\,
      S(0) => \alu_out0_carry__2_i_4_n_0\
    );
\alu_out0_carry__2_i_1\: unisim.vcomponents.LUT2
    generic map(
      INIT => X"6"
    )
        port map (
      I0 => alu_a(15),
      I1 => alu_b(15),
      O => \alu_out0_carry__2_i_1_n_0\
    );
\alu_out0_carry__2_i_2\: unisim.vcomponents.LUT2
    generic map(
      INIT => X"6"
    )
        port map (
      I0 => alu_a(14),
      I1 => alu_b(14),
      O => \alu_out0_carry__2_i_2_n_0\
    );
\alu_out0_carry__2_i_3\: unisim.vcomponents.LUT2
    generic map(
      INIT => X"6"
    )
        port map (
      I0 => alu_a(13),
      I1 => alu_b(13),
      O => \alu_out0_carry__2_i_3_n_0\
    );
\alu_out0_carry__2_i_4\: unisim.vcomponents.LUT2
    generic map(
      INIT => X"6"
    )
        port map (
      I0 => alu_a(12),
      I1 => alu_b(12),
      O => \alu_out0_carry__2_i_4_n_0\
    );
\alu_out0_carry__3\: unisim.vcomponents.CARRY4
     port map (
      CI => \alu_out0_carry__2_n_0\,
      CO(3) => \alu_out0_carry__3_n_0\,
      CO(2) => \alu_out0_carry__3_n_1\,
      CO(1) => \alu_out0_carry__3_n_2\,
      CO(0) => \alu_out0_carry__3_n_3\,
      CYINIT => '0',
      DI(3 downto 0) => alu_a(19 downto 16),
      O(3 downto 0) => data0(19 downto 16),
      S(3) => \alu_out0_carry__3_i_1_n_0\,
      S(2) => \alu_out0_carry__3_i_2_n_0\,
      S(1) => \alu_out0_carry__3_i_3_n_0\,
      S(0) => \alu_out0_carry__3_i_4_n_0\
    );
\alu_out0_carry__3_i_1\: unisim.vcomponents.LUT2
    generic map(
      INIT => X"6"
    )
        port map (
      I0 => alu_a(19),
      I1 => alu_b(19),
      O => \alu_out0_carry__3_i_1_n_0\
    );
\alu_out0_carry__3_i_2\: unisim.vcomponents.LUT2
    generic map(
      INIT => X"6"
    )
        port map (
      I0 => alu_a(18),
      I1 => alu_b(18),
      O => \alu_out0_carry__3_i_2_n_0\
    );
\alu_out0_carry__3_i_3\: unisim.vcomponents.LUT2
    generic map(
      INIT => X"6"
    )
        port map (
      I0 => alu_a(17),
      I1 => alu_b(17),
      O => \alu_out0_carry__3_i_3_n_0\
    );
\alu_out0_carry__3_i_4\: unisim.vcomponents.LUT2
    generic map(
      INIT => X"6"
    )
        port map (
      I0 => alu_a(16),
      I1 => alu_b(16),
      O => \alu_out0_carry__3_i_4_n_0\
    );
\alu_out0_carry__4\: unisim.vcomponents.CARRY4
     port map (
      CI => \alu_out0_carry__3_n_0\,
      CO(3) => \alu_out0_carry__4_n_0\,
      CO(2) => \alu_out0_carry__4_n_1\,
      CO(1) => \alu_out0_carry__4_n_2\,
      CO(0) => \alu_out0_carry__4_n_3\,
      CYINIT => '0',
      DI(3 downto 0) => alu_a(23 downto 20),
      O(3 downto 0) => data0(23 downto 20),
      S(3) => \alu_out0_carry__4_i_1_n_0\,
      S(2) => \alu_out0_carry__4_i_2_n_0\,
      S(1) => \alu_out0_carry__4_i_3_n_0\,
      S(0) => \alu_out0_carry__4_i_4_n_0\
    );
\alu_out0_carry__4_i_1\: unisim.vcomponents.LUT2
    generic map(
      INIT => X"6"
    )
        port map (
      I0 => alu_a(23),
      I1 => alu_b(23),
      O => \alu_out0_carry__4_i_1_n_0\
    );
\alu_out0_carry__4_i_2\: unisim.vcomponents.LUT2
    generic map(
      INIT => X"6"
    )
        port map (
      I0 => alu_a(22),
      I1 => alu_b(22),
      O => \alu_out0_carry__4_i_2_n_0\
    );
\alu_out0_carry__4_i_3\: unisim.vcomponents.LUT2
    generic map(
      INIT => X"6"
    )
        port map (
      I0 => alu_a(21),
      I1 => alu_b(21),
      O => \alu_out0_carry__4_i_3_n_0\
    );
\alu_out0_carry__4_i_4\: unisim.vcomponents.LUT2
    generic map(
      INIT => X"6"
    )
        port map (
      I0 => alu_a(20),
      I1 => alu_b(20),
      O => \alu_out0_carry__4_i_4_n_0\
    );
\alu_out0_carry__5\: unisim.vcomponents.CARRY4
     port map (
      CI => \alu_out0_carry__4_n_0\,
      CO(3) => \alu_out0_carry__5_n_0\,
      CO(2) => \alu_out0_carry__5_n_1\,
      CO(1) => \alu_out0_carry__5_n_2\,
      CO(0) => \alu_out0_carry__5_n_3\,
      CYINIT => '0',
      DI(3 downto 0) => alu_a(27 downto 24),
      O(3 downto 0) => data0(27 downto 24),
      S(3) => \alu_out0_carry__5_i_1_n_0\,
      S(2) => \alu_out0_carry__5_i_2_n_0\,
      S(1) => \alu_out0_carry__5_i_3_n_0\,
      S(0) => \alu_out0_carry__5_i_4_n_0\
    );
\alu_out0_carry__5_i_1\: unisim.vcomponents.LUT2
    generic map(
      INIT => X"6"
    )
        port map (
      I0 => alu_a(27),
      I1 => alu_b(27),
      O => \alu_out0_carry__5_i_1_n_0\
    );
\alu_out0_carry__5_i_2\: unisim.vcomponents.LUT2
    generic map(
      INIT => X"6"
    )
        port map (
      I0 => alu_a(26),
      I1 => alu_b(26),
      O => \alu_out0_carry__5_i_2_n_0\
    );
\alu_out0_carry__5_i_3\: unisim.vcomponents.LUT2
    generic map(
      INIT => X"6"
    )
        port map (
      I0 => alu_a(25),
      I1 => alu_b(25),
      O => \alu_out0_carry__5_i_3_n_0\
    );
\alu_out0_carry__5_i_4\: unisim.vcomponents.LUT2
    generic map(
      INIT => X"6"
    )
        port map (
      I0 => alu_a(24),
      I1 => alu_b(24),
      O => \alu_out0_carry__5_i_4_n_0\
    );
\alu_out0_carry__6\: unisim.vcomponents.CARRY4
     port map (
      CI => \alu_out0_carry__5_n_0\,
      CO(3) => \NLW_alu_out0_carry__6_CO_UNCONNECTED\(3),
      CO(2) => \alu_out0_carry__6_n_1\,
      CO(1) => \alu_out0_carry__6_n_2\,
      CO(0) => \alu_out0_carry__6_n_3\,
      CYINIT => '0',
      DI(3) => '0',
      DI(2 downto 0) => alu_a(30 downto 28),
      O(3 downto 0) => data0(31 downto 28),
      S(3) => \alu_out0_carry__6_i_1_n_0\,
      S(2) => \alu_out0_carry__6_i_2_n_0\,
      S(1) => \alu_out0_carry__6_i_3_n_0\,
      S(0) => \alu_out0_carry__6_i_4_n_0\
    );
\alu_out0_carry__6_i_1\: unisim.vcomponents.LUT2
    generic map(
      INIT => X"6"
    )
        port map (
      I0 => alu_b(31),
      I1 => alu_a(31),
      O => \alu_out0_carry__6_i_1_n_0\
    );
\alu_out0_carry__6_i_2\: unisim.vcomponents.LUT2
    generic map(
      INIT => X"6"
    )
        port map (
      I0 => alu_a(30),
      I1 => alu_b(30),
      O => \alu_out0_carry__6_i_2_n_0\
    );
\alu_out0_carry__6_i_3\: unisim.vcomponents.LUT2
    generic map(
      INIT => X"6"
    )
        port map (
      I0 => alu_a(29),
      I1 => alu_b(29),
      O => \alu_out0_carry__6_i_3_n_0\
    );
\alu_out0_carry__6_i_4\: unisim.vcomponents.LUT2
    generic map(
      INIT => X"6"
    )
        port map (
      I0 => alu_a(28),
      I1 => alu_b(28),
      O => \alu_out0_carry__6_i_4_n_0\
    );
alu_out0_carry_i_1: unisim.vcomponents.LUT2
    generic map(
      INIT => X"6"
    )
        port map (
      I0 => alu_a(3),
      I1 => alu_b(3),
      O => alu_out0_carry_i_1_n_0
    );
alu_out0_carry_i_2: unisim.vcomponents.LUT2
    generic map(
      INIT => X"6"
    )
        port map (
      I0 => alu_a(2),
      I1 => alu_b(2),
      O => alu_out0_carry_i_2_n_0
    );
alu_out0_carry_i_3: unisim.vcomponents.LUT2
    generic map(
      INIT => X"6"
    )
        port map (
      I0 => alu_a(1),
      I1 => alu_b(1),
      O => alu_out0_carry_i_3_n_0
    );
alu_out0_carry_i_4: unisim.vcomponents.LUT2
    generic map(
      INIT => X"6"
    )
        port map (
      I0 => alu_a(0),
      I1 => alu_b(0),
      O => alu_out0_carry_i_4_n_0
    );
\alu_out[0]_INST_0\: unisim.vcomponents.LUT5
    generic map(
      INIT => X"FFFF4540"
    )
        port map (
      I0 => alu_sel(2),
      I1 => alu_out_0_sn_1,
      I2 => alu_sel(1),
      I3 => \alu_out[0]_INST_0_i_2_n_0\,
      I4 => \alu_out[0]_INST_0_i_3_n_0\,
      O => alu_out(0)
    );
\alu_out[0]_INST_0_i_2\: unisim.vcomponents.LUT4
    generic map(
      INIT => X"8F80"
    )
        port map (
      I0 => alu_b(0),
      I1 => alu_a(0),
      I2 => alu_sel(0),
      I3 => data0(0),
      O => \alu_out[0]_INST_0_i_2_n_0\
    );
\alu_out[0]_INST_0_i_3\: unisim.vcomponents.LUT6
    generic map(
      INIT => X"006F000000600000"
    )
        port map (
      I0 => alu_a(0),
      I1 => alu_b(0),
      I2 => alu_sel(0),
      I3 => alu_sel(1),
      I4 => alu_sel(2),
      I5 => data4,
      O => \alu_out[0]_INST_0_i_3_n_0\
    );
\alu_out[10]_INST_0\: unisim.vcomponents.LUT6
    generic map(
      INIT => X"00000000FEAEAAAA"
    )
        port map (
      I0 => alu_out_10_sn_1,
      I1 => \alu_out[9]_0\,
      I2 => alu_b(0),
      I3 => \alu_out[11]_0\,
      I4 => alu_sel(0),
      I5 => \alu_out[10]_INST_0_i_3_n_0\,
      O => alu_out(10)
    );
\alu_out[10]_INST_0_i_3\: unisim.vcomponents.LUT6
    generic map(
      INIT => X"FFFF99FF0000770F"
    )
        port map (
      I0 => alu_a(10),
      I1 => alu_b(10),
      I2 => data0(10),
      I3 => alu_sel(0),
      I4 => alu_sel(1),
      I5 => alu_sel(2),
      O => \alu_out[10]_INST_0_i_3_n_0\
    );
\alu_out[11]_INST_0\: unisim.vcomponents.LUT6
    generic map(
      INIT => X"00000000FEAEAAAA"
    )
        port map (
      I0 => alu_out_11_sn_1,
      I1 => \alu_out[11]_0\,
      I2 => alu_b(0),
      I3 => \alu_out[11]_1\,
      I4 => alu_sel(0),
      I5 => \alu_out[11]_INST_0_i_3_n_0\,
      O => alu_out(11)
    );
\alu_out[11]_INST_0_i_3\: unisim.vcomponents.LUT6
    generic map(
      INIT => X"FFFF99FF0000770F"
    )
        port map (
      I0 => alu_a(11),
      I1 => alu_b(11),
      I2 => data0(11),
      I3 => alu_sel(0),
      I4 => alu_sel(1),
      I5 => alu_sel(2),
      O => \alu_out[11]_INST_0_i_3_n_0\
    );
\alu_out[12]_INST_0\: unisim.vcomponents.LUT6
    generic map(
      INIT => X"00000000FEAEAAAA"
    )
        port map (
      I0 => alu_out_12_sn_1,
      I1 => \alu_out[11]_1\,
      I2 => alu_b(0),
      I3 => \alu_out[13]_0\,
      I4 => alu_sel(0),
      I5 => \alu_out[12]_INST_0_i_3_n_0\,
      O => alu_out(12)
    );
\alu_out[12]_INST_0_i_3\: unisim.vcomponents.LUT6
    generic map(
      INIT => X"FFFF99FF0000770F"
    )
        port map (
      I0 => alu_a(12),
      I1 => alu_b(12),
      I2 => data0(12),
      I3 => alu_sel(0),
      I4 => alu_sel(1),
      I5 => alu_sel(2),
      O => \alu_out[12]_INST_0_i_3_n_0\
    );
\alu_out[13]_INST_0\: unisim.vcomponents.LUT6
    generic map(
      INIT => X"00000000FEAEAAAA"
    )
        port map (
      I0 => alu_out_13_sn_1,
      I1 => \alu_out[13]_0\,
      I2 => alu_b(0),
      I3 => \alu_out[13]_1\,
      I4 => alu_sel(0),
      I5 => \alu_out[13]_INST_0_i_3_n_0\,
      O => alu_out(13)
    );
\alu_out[13]_INST_0_i_3\: unisim.vcomponents.LUT6
    generic map(
      INIT => X"FFFF99FF0000770F"
    )
        port map (
      I0 => alu_a(13),
      I1 => alu_b(13),
      I2 => data0(13),
      I3 => alu_sel(0),
      I4 => alu_sel(1),
      I5 => alu_sel(2),
      O => \alu_out[13]_INST_0_i_3_n_0\
    );
\alu_out[14]_INST_0\: unisim.vcomponents.LUT6
    generic map(
      INIT => X"00000000FEAEAAAA"
    )
        port map (
      I0 => alu_out_14_sn_1,
      I1 => \alu_out[13]_1\,
      I2 => alu_b(0),
      I3 => \alu_out[15]_0\,
      I4 => alu_sel(0),
      I5 => \alu_out[14]_INST_0_i_3_n_0\,
      O => alu_out(14)
    );
\alu_out[14]_INST_0_i_3\: unisim.vcomponents.LUT6
    generic map(
      INIT => X"FFFF99FF0000770F"
    )
        port map (
      I0 => alu_a(14),
      I1 => alu_b(14),
      I2 => data0(14),
      I3 => alu_sel(0),
      I4 => alu_sel(1),
      I5 => alu_sel(2),
      O => \alu_out[14]_INST_0_i_3_n_0\
    );
\alu_out[15]_INST_0\: unisim.vcomponents.LUT6
    generic map(
      INIT => X"00000000FEAEAAAA"
    )
        port map (
      I0 => alu_out_15_sn_1,
      I1 => \alu_out[15]_0\,
      I2 => alu_b(0),
      I3 => \alu_out[15]_1\,
      I4 => alu_sel(0),
      I5 => \alu_out[15]_INST_0_i_3_n_0\,
      O => alu_out(15)
    );
\alu_out[15]_INST_0_i_3\: unisim.vcomponents.LUT6
    generic map(
      INIT => X"FFFF99FF0000770F"
    )
        port map (
      I0 => alu_a(15),
      I1 => alu_b(15),
      I2 => data0(15),
      I3 => alu_sel(0),
      I4 => alu_sel(1),
      I5 => alu_sel(2),
      O => \alu_out[15]_INST_0_i_3_n_0\
    );
\alu_out[16]_INST_0\: unisim.vcomponents.LUT6
    generic map(
      INIT => X"00000000FEAEAAAA"
    )
        port map (
      I0 => alu_out_16_sn_1,
      I1 => \alu_out[15]_1\,
      I2 => alu_b(0),
      I3 => \alu_out[17]_0\,
      I4 => alu_sel(0),
      I5 => \alu_out[16]_INST_0_i_3_n_0\,
      O => alu_out(16)
    );
\alu_out[16]_INST_0_i_3\: unisim.vcomponents.LUT6
    generic map(
      INIT => X"FFFF99FF0000770F"
    )
        port map (
      I0 => alu_a(16),
      I1 => alu_b(16),
      I2 => data0(16),
      I3 => alu_sel(0),
      I4 => alu_sel(1),
      I5 => alu_sel(2),
      O => \alu_out[16]_INST_0_i_3_n_0\
    );
\alu_out[17]_INST_0\: unisim.vcomponents.LUT6
    generic map(
      INIT => X"00000000FEAEAAAA"
    )
        port map (
      I0 => alu_out_17_sn_1,
      I1 => \alu_out[17]_0\,
      I2 => alu_b(0),
      I3 => \alu_out[17]_1\,
      I4 => alu_sel(0),
      I5 => \alu_out[17]_INST_0_i_3_n_0\,
      O => alu_out(17)
    );
\alu_out[17]_INST_0_i_3\: unisim.vcomponents.LUT6
    generic map(
      INIT => X"FFFF99FF0000770F"
    )
        port map (
      I0 => alu_a(17),
      I1 => alu_b(17),
      I2 => data0(17),
      I3 => alu_sel(0),
      I4 => alu_sel(1),
      I5 => alu_sel(2),
      O => \alu_out[17]_INST_0_i_3_n_0\
    );
\alu_out[18]_INST_0\: unisim.vcomponents.LUT6
    generic map(
      INIT => X"00000000FEAEAAAA"
    )
        port map (
      I0 => alu_out_18_sn_1,
      I1 => \alu_out[17]_1\,
      I2 => alu_b(0),
      I3 => \alu_out[18]_0\,
      I4 => alu_sel(0),
      I5 => \alu_out[18]_INST_0_i_3_n_0\,
      O => alu_out(18)
    );
\alu_out[18]_INST_0_i_3\: unisim.vcomponents.LUT6
    generic map(
      INIT => X"FFFF99FF0000770F"
    )
        port map (
      I0 => alu_a(18),
      I1 => alu_b(18),
      I2 => data0(18),
      I3 => alu_sel(0),
      I4 => alu_sel(1),
      I5 => alu_sel(2),
      O => \alu_out[18]_INST_0_i_3_n_0\
    );
\alu_out[19]_INST_0\: unisim.vcomponents.LUT6
    generic map(
      INIT => X"00000000FEAEAAAA"
    )
        port map (
      I0 => alu_out_19_sn_1,
      I1 => \alu_out[18]_0\,
      I2 => alu_b(0),
      I3 => \alu_out[19]_0\,
      I4 => alu_sel(0),
      I5 => \alu_out[19]_INST_0_i_3_n_0\,
      O => alu_out(19)
    );
\alu_out[19]_INST_0_i_3\: unisim.vcomponents.LUT6
    generic map(
      INIT => X"FFFF99FF0000770F"
    )
        port map (
      I0 => alu_a(19),
      I1 => alu_b(19),
      I2 => data0(19),
      I3 => alu_sel(0),
      I4 => alu_sel(1),
      I5 => alu_sel(2),
      O => \alu_out[19]_INST_0_i_3_n_0\
    );
\alu_out[1]_INST_0\: unisim.vcomponents.LUT6
    generic map(
      INIT => X"00000000FEAEAAAA"
    )
        port map (
      I0 => alu_out_1_sn_1,
      I1 => \alu_out[1]_0\,
      I2 => alu_b(0),
      I3 => \alu_out[1]_1\,
      I4 => alu_sel(0),
      I5 => \alu_out[1]_INST_0_i_3_n_0\,
      O => alu_out(1)
    );
\alu_out[1]_INST_0_i_3\: unisim.vcomponents.LUT6
    generic map(
      INIT => X"FFFF99FF0000770F"
    )
        port map (
      I0 => alu_b(1),
      I1 => alu_a(1),
      I2 => data0(1),
      I3 => alu_sel(0),
      I4 => alu_sel(1),
      I5 => alu_sel(2),
      O => \alu_out[1]_INST_0_i_3_n_0\
    );
\alu_out[20]_INST_0\: unisim.vcomponents.LUT6
    generic map(
      INIT => X"00000000FEAEAAAA"
    )
        port map (
      I0 => alu_out_20_sn_1,
      I1 => \alu_out[19]_0\,
      I2 => alu_b(0),
      I3 => \alu_out[20]_0\,
      I4 => alu_sel(0),
      I5 => \alu_out[20]_INST_0_i_3_n_0\,
      O => alu_out(20)
    );
\alu_out[20]_INST_0_i_3\: unisim.vcomponents.LUT6
    generic map(
      INIT => X"FFFF99FF0000770F"
    )
        port map (
      I0 => alu_a(20),
      I1 => alu_b(20),
      I2 => data0(20),
      I3 => alu_sel(0),
      I4 => alu_sel(1),
      I5 => alu_sel(2),
      O => \alu_out[20]_INST_0_i_3_n_0\
    );
\alu_out[21]_INST_0\: unisim.vcomponents.LUT6
    generic map(
      INIT => X"00000000FEAEAAAA"
    )
        port map (
      I0 => alu_out_21_sn_1,
      I1 => \alu_out[20]_0\,
      I2 => alu_b(0),
      I3 => \alu_out[21]_0\,
      I4 => alu_sel(0),
      I5 => \alu_out[21]_INST_0_i_3_n_0\,
      O => alu_out(21)
    );
\alu_out[21]_INST_0_i_3\: unisim.vcomponents.LUT6
    generic map(
      INIT => X"FFFF99FF0000770F"
    )
        port map (
      I0 => alu_a(21),
      I1 => alu_b(21),
      I2 => data0(21),
      I3 => alu_sel(0),
      I4 => alu_sel(1),
      I5 => alu_sel(2),
      O => \alu_out[21]_INST_0_i_3_n_0\
    );
\alu_out[22]_INST_0\: unisim.vcomponents.LUT6
    generic map(
      INIT => X"00000000FEAEAAAA"
    )
        port map (
      I0 => alu_out_22_sn_1,
      I1 => \alu_out[21]_0\,
      I2 => alu_b(0),
      I3 => \alu_out[22]_0\,
      I4 => alu_sel(0),
      I5 => \alu_out[22]_INST_0_i_3_n_0\,
      O => alu_out(22)
    );
\alu_out[22]_INST_0_i_3\: unisim.vcomponents.LUT6
    generic map(
      INIT => X"FFFF99FF0000770F"
    )
        port map (
      I0 => alu_a(22),
      I1 => alu_b(22),
      I2 => data0(22),
      I3 => alu_sel(0),
      I4 => alu_sel(1),
      I5 => alu_sel(2),
      O => \alu_out[22]_INST_0_i_3_n_0\
    );
\alu_out[23]_INST_0\: unisim.vcomponents.LUT6
    generic map(
      INIT => X"00000000FEAEAAAA"
    )
        port map (
      I0 => alu_out_23_sn_1,
      I1 => \alu_out[22]_0\,
      I2 => alu_b(0),
      I3 => \alu_out[23]_0\,
      I4 => alu_sel(0),
      I5 => \alu_out[23]_INST_0_i_3_n_0\,
      O => alu_out(23)
    );
\alu_out[23]_INST_0_i_3\: unisim.vcomponents.LUT6
    generic map(
      INIT => X"FFFF99FF0000770F"
    )
        port map (
      I0 => alu_a(23),
      I1 => alu_b(23),
      I2 => data0(23),
      I3 => alu_sel(0),
      I4 => alu_sel(1),
      I5 => alu_sel(2),
      O => \alu_out[23]_INST_0_i_3_n_0\
    );
\alu_out[24]_INST_0\: unisim.vcomponents.LUT6
    generic map(
      INIT => X"00000000FEAEAAAA"
    )
        port map (
      I0 => alu_out_24_sn_1,
      I1 => \alu_out[23]_0\,
      I2 => alu_b(0),
      I3 => \alu_out[24]_0\,
      I4 => alu_sel(0),
      I5 => \alu_out[24]_INST_0_i_3_n_0\,
      O => alu_out(24)
    );
\alu_out[24]_INST_0_i_3\: unisim.vcomponents.LUT6
    generic map(
      INIT => X"FFFF99FF0000770F"
    )
        port map (
      I0 => alu_a(24),
      I1 => alu_b(24),
      I2 => data0(24),
      I3 => alu_sel(0),
      I4 => alu_sel(1),
      I5 => alu_sel(2),
      O => \alu_out[24]_INST_0_i_3_n_0\
    );
\alu_out[25]_INST_0\: unisim.vcomponents.LUT6
    generic map(
      INIT => X"00000000FEAEAAAA"
    )
        port map (
      I0 => alu_out_25_sn_1,
      I1 => \alu_out[24]_0\,
      I2 => alu_b(0),
      I3 => \alu_out[25]_0\,
      I4 => alu_sel(0),
      I5 => \alu_out[25]_INST_0_i_3_n_0\,
      O => alu_out(25)
    );
\alu_out[25]_INST_0_i_3\: unisim.vcomponents.LUT6
    generic map(
      INIT => X"FFFF99FF0000770F"
    )
        port map (
      I0 => alu_a(25),
      I1 => alu_b(25),
      I2 => data0(25),
      I3 => alu_sel(0),
      I4 => alu_sel(1),
      I5 => alu_sel(2),
      O => \alu_out[25]_INST_0_i_3_n_0\
    );
\alu_out[26]_INST_0\: unisim.vcomponents.LUT6
    generic map(
      INIT => X"00000000FEAEAAAA"
    )
        port map (
      I0 => alu_out_26_sn_1,
      I1 => \alu_out[25]_0\,
      I2 => alu_b(0),
      I3 => \alu_out[26]_0\,
      I4 => alu_sel(0),
      I5 => \alu_out[26]_INST_0_i_3_n_0\,
      O => alu_out(26)
    );
\alu_out[26]_INST_0_i_3\: unisim.vcomponents.LUT6
    generic map(
      INIT => X"FFFF99FF0000770F"
    )
        port map (
      I0 => alu_a(26),
      I1 => alu_b(26),
      I2 => data0(26),
      I3 => alu_sel(0),
      I4 => alu_sel(1),
      I5 => alu_sel(2),
      O => \alu_out[26]_INST_0_i_3_n_0\
    );
\alu_out[27]_INST_0\: unisim.vcomponents.LUT6
    generic map(
      INIT => X"00000000FEAEAAAA"
    )
        port map (
      I0 => alu_out_27_sn_1,
      I1 => \alu_out[26]_0\,
      I2 => alu_b(0),
      I3 => \alu_out[27]_0\,
      I4 => alu_sel(0),
      I5 => \alu_out[27]_INST_0_i_3_n_0\,
      O => alu_out(27)
    );
\alu_out[27]_INST_0_i_3\: unisim.vcomponents.LUT6
    generic map(
      INIT => X"FFFF99FF0000770F"
    )
        port map (
      I0 => alu_a(27),
      I1 => alu_b(27),
      I2 => data0(27),
      I3 => alu_sel(0),
      I4 => alu_sel(1),
      I5 => alu_sel(2),
      O => \alu_out[27]_INST_0_i_3_n_0\
    );
\alu_out[28]_INST_0\: unisim.vcomponents.LUT6
    generic map(
      INIT => X"00000000FEAEAAAA"
    )
        port map (
      I0 => alu_out_28_sn_1,
      I1 => \alu_out[27]_0\,
      I2 => alu_b(0),
      I3 => \alu_out[28]_0\,
      I4 => alu_sel(0),
      I5 => \alu_out[28]_INST_0_i_3_n_0\,
      O => alu_out(28)
    );
\alu_out[28]_INST_0_i_3\: unisim.vcomponents.LUT6
    generic map(
      INIT => X"FFFF99FF0000770F"
    )
        port map (
      I0 => alu_a(28),
      I1 => alu_b(28),
      I2 => data0(28),
      I3 => alu_sel(0),
      I4 => alu_sel(1),
      I5 => alu_sel(2),
      O => \alu_out[28]_INST_0_i_3_n_0\
    );
\alu_out[29]_INST_0\: unisim.vcomponents.LUT6
    generic map(
      INIT => X"00000000FEAEAAAA"
    )
        port map (
      I0 => alu_out_29_sn_1,
      I1 => \alu_out[28]_0\,
      I2 => alu_b(0),
      I3 => \alu_out[29]_0\,
      I4 => alu_sel(0),
      I5 => \alu_out[29]_INST_0_i_3_n_0\,
      O => alu_out(29)
    );
\alu_out[29]_INST_0_i_3\: unisim.vcomponents.LUT6
    generic map(
      INIT => X"FFFF99FF0000770F"
    )
        port map (
      I0 => alu_a(29),
      I1 => alu_b(29),
      I2 => data0(29),
      I3 => alu_sel(0),
      I4 => alu_sel(1),
      I5 => alu_sel(2),
      O => \alu_out[29]_INST_0_i_3_n_0\
    );
\alu_out[2]_INST_0\: unisim.vcomponents.LUT6
    generic map(
      INIT => X"00000000FEAEAAAA"
    )
        port map (
      I0 => alu_out_2_sn_1,
      I1 => \alu_out[1]_1\,
      I2 => alu_b(0),
      I3 => \alu_out[2]_0\,
      I4 => alu_sel(0),
      I5 => \alu_out[2]_INST_0_i_3_n_0\,
      O => alu_out(2)
    );
\alu_out[2]_INST_0_i_3\: unisim.vcomponents.LUT6
    generic map(
      INIT => X"FFFF99FF0000770F"
    )
        port map (
      I0 => alu_b(2),
      I1 => alu_a(2),
      I2 => data0(2),
      I3 => alu_sel(0),
      I4 => alu_sel(1),
      I5 => alu_sel(2),
      O => \alu_out[2]_INST_0_i_3_n_0\
    );
\alu_out[30]_INST_0\: unisim.vcomponents.LUT6
    generic map(
      INIT => X"00000000FEAEAAAA"
    )
        port map (
      I0 => alu_out_30_sn_1,
      I1 => \alu_out[29]_0\,
      I2 => alu_b(0),
      I3 => alu_a(31),
      I4 => alu_sel(0),
      I5 => \alu_out[30]_INST_0_i_3_n_0\,
      O => alu_out(30)
    );
\alu_out[30]_INST_0_i_3\: unisim.vcomponents.LUT6
    generic map(
      INIT => X"FFFF99FF0000770F"
    )
        port map (
      I0 => alu_b(30),
      I1 => alu_a(30),
      I2 => data0(30),
      I3 => alu_sel(0),
      I4 => alu_sel(1),
      I5 => alu_sel(2),
      O => \alu_out[30]_INST_0_i_3_n_0\
    );
\alu_out[31]_INST_0\: unisim.vcomponents.LUT6
    generic map(
      INIT => X"00000000FFFBFBFB"
    )
        port map (
      I0 => alu_out_31_sn_1,
      I1 => alu_sel(1),
      I2 => \alu_out[31]_0\,
      I3 => \alu_out[31]_1\,
      I4 => \alu_out[31]_2\,
      I5 => \alu_out[31]_INST_0_i_5_n_0\,
      O => alu_out(31)
    );
\alu_out[31]_INST_0_i_5\: unisim.vcomponents.LUT6
    generic map(
      INIT => X"FFC3FFFF003F0055"
    )
        port map (
      I0 => data0(31),
      I1 => alu_a(31),
      I2 => alu_b(31),
      I3 => alu_sel(1),
      I4 => alu_sel(0),
      I5 => alu_sel(2),
      O => \alu_out[31]_INST_0_i_5_n_0\
    );
\alu_out[3]_INST_0\: unisim.vcomponents.LUT6
    generic map(
      INIT => X"00000000FEAEAAAA"
    )
        port map (
      I0 => alu_out_3_sn_1,
      I1 => \alu_out[2]_0\,
      I2 => alu_b(0),
      I3 => \alu_out[4]_0\,
      I4 => alu_sel(0),
      I5 => \alu_out[3]_INST_0_i_3_n_0\,
      O => alu_out(3)
    );
\alu_out[3]_INST_0_i_3\: unisim.vcomponents.LUT6
    generic map(
      INIT => X"F0C3C3F3F1F1F1F1"
    )
        port map (
      I0 => data0(3),
      I1 => alu_sel(1),
      I2 => alu_sel(2),
      I3 => alu_a(3),
      I4 => alu_b(3),
      I5 => alu_sel(0),
      O => \alu_out[3]_INST_0_i_3_n_0\
    );
\alu_out[4]_INST_0\: unisim.vcomponents.LUT6
    generic map(
      INIT => X"00000000FEAEAAAA"
    )
        port map (
      I0 => alu_out_4_sn_1,
      I1 => \alu_out[4]_0\,
      I2 => alu_b(0),
      I3 => \alu_out[4]_1\,
      I4 => alu_sel(0),
      I5 => \alu_out[4]_INST_0_i_3_n_0\,
      O => alu_out(4)
    );
\alu_out[4]_INST_0_i_3\: unisim.vcomponents.LUT6
    generic map(
      INIT => X"FFFF99FF0000770F"
    )
        port map (
      I0 => alu_b(4),
      I1 => alu_a(4),
      I2 => data0(4),
      I3 => alu_sel(0),
      I4 => alu_sel(1),
      I5 => alu_sel(2),
      O => \alu_out[4]_INST_0_i_3_n_0\
    );
\alu_out[5]_INST_0\: unisim.vcomponents.LUT6
    generic map(
      INIT => X"00000000FEAEAAAA"
    )
        port map (
      I0 => alu_out_5_sn_1,
      I1 => \alu_out[4]_1\,
      I2 => alu_b(0),
      I3 => \alu_out[5]_0\,
      I4 => alu_sel(0),
      I5 => \alu_out[5]_INST_0_i_3_n_0\,
      O => alu_out(5)
    );
\alu_out[5]_INST_0_i_3\: unisim.vcomponents.LUT6
    generic map(
      INIT => X"FFFF99FF0000770F"
    )
        port map (
      I0 => alu_a(5),
      I1 => alu_b(5),
      I2 => data0(5),
      I3 => alu_sel(0),
      I4 => alu_sel(1),
      I5 => alu_sel(2),
      O => \alu_out[5]_INST_0_i_3_n_0\
    );
\alu_out[6]_INST_0\: unisim.vcomponents.LUT6
    generic map(
      INIT => X"00000000FEAEAAAA"
    )
        port map (
      I0 => alu_out_6_sn_1,
      I1 => \alu_out[5]_0\,
      I2 => alu_b(0),
      I3 => \alu_out[6]_0\,
      I4 => alu_sel(0),
      I5 => \alu_out[6]_INST_0_i_3_n_0\,
      O => alu_out(6)
    );
\alu_out[6]_INST_0_i_3\: unisim.vcomponents.LUT6
    generic map(
      INIT => X"FFFF99FF0000770F"
    )
        port map (
      I0 => alu_a(6),
      I1 => alu_b(6),
      I2 => data0(6),
      I3 => alu_sel(0),
      I4 => alu_sel(1),
      I5 => alu_sel(2),
      O => \alu_out[6]_INST_0_i_3_n_0\
    );
\alu_out[7]_INST_0\: unisim.vcomponents.LUT6
    generic map(
      INIT => X"00000000FEAEAAAA"
    )
        port map (
      I0 => alu_out_7_sn_1,
      I1 => \alu_out[6]_0\,
      I2 => alu_b(0),
      I3 => \alu_out[7]_0\,
      I4 => alu_sel(0),
      I5 => \alu_out[7]_INST_0_i_3_n_0\,
      O => alu_out(7)
    );
\alu_out[7]_INST_0_i_3\: unisim.vcomponents.LUT6
    generic map(
      INIT => X"FFFF99FF0000770F"
    )
        port map (
      I0 => alu_a(7),
      I1 => alu_b(7),
      I2 => data0(7),
      I3 => alu_sel(0),
      I4 => alu_sel(1),
      I5 => alu_sel(2),
      O => \alu_out[7]_INST_0_i_3_n_0\
    );
\alu_out[8]_INST_0\: unisim.vcomponents.LUT6
    generic map(
      INIT => X"00000000FEAEAAAA"
    )
        port map (
      I0 => alu_out_8_sn_1,
      I1 => \alu_out[7]_0\,
      I2 => alu_b(0),
      I3 => \alu_out[8]_0\,
      I4 => alu_sel(0),
      I5 => \alu_out[8]_INST_0_i_3_n_0\,
      O => alu_out(8)
    );
\alu_out[8]_INST_0_i_3\: unisim.vcomponents.LUT6
    generic map(
      INIT => X"FFFF99FF0000770F"
    )
        port map (
      I0 => alu_a(8),
      I1 => alu_b(8),
      I2 => data0(8),
      I3 => alu_sel(0),
      I4 => alu_sel(1),
      I5 => alu_sel(2),
      O => \alu_out[8]_INST_0_i_3_n_0\
    );
\alu_out[9]_INST_0\: unisim.vcomponents.LUT6
    generic map(
      INIT => X"00000000FEAEAAAA"
    )
        port map (
      I0 => alu_out_9_sn_1,
      I1 => \alu_out[8]_0\,
      I2 => alu_b(0),
      I3 => \alu_out[9]_0\,
      I4 => alu_sel(0),
      I5 => \alu_out[9]_INST_0_i_3_n_0\,
      O => alu_out(9)
    );
\alu_out[9]_INST_0_i_3\: unisim.vcomponents.LUT6
    generic map(
      INIT => X"FFFF99FF0000770F"
    )
        port map (
      I0 => alu_a(9),
      I1 => alu_b(9),
      I2 => data0(9),
      I3 => alu_sel(0),
      I4 => alu_sel(1),
      I5 => alu_sel(2),
      O => \alu_out[9]_INST_0_i_3_n_0\
    );
end STRUCTURE;
library IEEE;
use IEEE.STD_LOGIC_1164.ALL;
library UNISIM;
use UNISIM.VCOMPONENTS.ALL;
entity decalper_eb_ot_sdeen_pot_pi_dehcac_xnilix is
  port (
    alu_a : in STD_LOGIC_VECTOR ( 31 downto 0 );
    alu_b : in STD_LOGIC_VECTOR ( 31 downto 0 );
    alu_sel : in STD_LOGIC_VECTOR ( 2 downto 0 );
    alu_out : out STD_LOGIC_VECTOR ( 31 downto 0 )
  );
  attribute NotValidForBitStream : boolean;
  attribute NotValidForBitStream of decalper_eb_ot_sdeen_pot_pi_dehcac_xnilix : entity is true;
  attribute CHECK_LICENSE_TYPE : string;
  attribute CHECK_LICENSE_TYPE of decalper_eb_ot_sdeen_pot_pi_dehcac_xnilix : entity is "risc32_alu32_0_0,alu32,{}";
  attribute DowngradeIPIdentifiedWarnings : string;
  attribute DowngradeIPIdentifiedWarnings of decalper_eb_ot_sdeen_pot_pi_dehcac_xnilix : entity is "yes";
  attribute IP_DEFINITION_SOURCE : string;
  attribute IP_DEFINITION_SOURCE of decalper_eb_ot_sdeen_pot_pi_dehcac_xnilix : entity is "module_ref";
  attribute X_CORE_INFO : string;
  attribute X_CORE_INFO of decalper_eb_ot_sdeen_pot_pi_dehcac_xnilix : entity is "alu32,Vivado 2022.2";
end decalper_eb_ot_sdeen_pot_pi_dehcac_xnilix;

architecture STRUCTURE of decalper_eb_ot_sdeen_pot_pi_dehcac_xnilix is
  signal \alu_out[0]_INST_0_i_1_n_0\ : STD_LOGIC;
  signal \alu_out[0]_INST_0_i_4_n_0\ : STD_LOGIC;
  signal \alu_out[0]_INST_0_i_5_n_0\ : STD_LOGIC;
  signal \alu_out[10]_INST_0_i_1_n_0\ : STD_LOGIC;
  signal \alu_out[10]_INST_0_i_2_n_0\ : STD_LOGIC;
  signal \alu_out[10]_INST_0_i_4_n_0\ : STD_LOGIC;
  signal \alu_out[10]_INST_0_i_5_n_0\ : STD_LOGIC;
  signal \alu_out[10]_INST_0_i_6_n_0\ : STD_LOGIC;
  signal \alu_out[11]_INST_0_i_1_n_0\ : STD_LOGIC;
  signal \alu_out[11]_INST_0_i_2_n_0\ : STD_LOGIC;
  signal \alu_out[11]_INST_0_i_4_n_0\ : STD_LOGIC;
  signal \alu_out[11]_INST_0_i_5_n_0\ : STD_LOGIC;
  signal \alu_out[11]_INST_0_i_6_n_0\ : STD_LOGIC;
  signal \alu_out[12]_INST_0_i_1_n_0\ : STD_LOGIC;
  signal \alu_out[12]_INST_0_i_2_n_0\ : STD_LOGIC;
  signal \alu_out[12]_INST_0_i_4_n_0\ : STD_LOGIC;
  signal \alu_out[12]_INST_0_i_5_n_0\ : STD_LOGIC;
  signal \alu_out[12]_INST_0_i_6_n_0\ : STD_LOGIC;
  signal \alu_out[13]_INST_0_i_1_n_0\ : STD_LOGIC;
  signal \alu_out[13]_INST_0_i_2_n_0\ : STD_LOGIC;
  signal \alu_out[13]_INST_0_i_4_n_0\ : STD_LOGIC;
  signal \alu_out[13]_INST_0_i_5_n_0\ : STD_LOGIC;
  signal \alu_out[13]_INST_0_i_6_n_0\ : STD_LOGIC;
  signal \alu_out[14]_INST_0_i_1_n_0\ : STD_LOGIC;
  signal \alu_out[14]_INST_0_i_2_n_0\ : STD_LOGIC;
  signal \alu_out[14]_INST_0_i_4_n_0\ : STD_LOGIC;
  signal \alu_out[14]_INST_0_i_5_n_0\ : STD_LOGIC;
  signal \alu_out[14]_INST_0_i_6_n_0\ : STD_LOGIC;
  signal \alu_out[15]_INST_0_i_1_n_0\ : STD_LOGIC;
  signal \alu_out[15]_INST_0_i_2_n_0\ : STD_LOGIC;
  signal \alu_out[15]_INST_0_i_4_n_0\ : STD_LOGIC;
  signal \alu_out[15]_INST_0_i_5_n_0\ : STD_LOGIC;
  signal \alu_out[15]_INST_0_i_6_n_0\ : STD_LOGIC;
  signal \alu_out[15]_INST_0_i_7_n_0\ : STD_LOGIC;
  signal \alu_out[16]_INST_0_i_1_n_0\ : STD_LOGIC;
  signal \alu_out[16]_INST_0_i_2_n_0\ : STD_LOGIC;
  signal \alu_out[16]_INST_0_i_4_n_0\ : STD_LOGIC;
  signal \alu_out[16]_INST_0_i_5_n_0\ : STD_LOGIC;
  signal \alu_out[16]_INST_0_i_6_n_0\ : STD_LOGIC;
  signal \alu_out[16]_INST_0_i_7_n_0\ : STD_LOGIC;
  signal \alu_out[17]_INST_0_i_1_n_0\ : STD_LOGIC;
  signal \alu_out[17]_INST_0_i_2_n_0\ : STD_LOGIC;
  signal \alu_out[17]_INST_0_i_4_n_0\ : STD_LOGIC;
  signal \alu_out[17]_INST_0_i_5_n_0\ : STD_LOGIC;
  signal \alu_out[17]_INST_0_i_6_n_0\ : STD_LOGIC;
  signal \alu_out[17]_INST_0_i_7_n_0\ : STD_LOGIC;
  signal \alu_out[18]_INST_0_i_1_n_0\ : STD_LOGIC;
  signal \alu_out[18]_INST_0_i_2_n_0\ : STD_LOGIC;
  signal \alu_out[18]_INST_0_i_4_n_0\ : STD_LOGIC;
  signal \alu_out[18]_INST_0_i_5_n_0\ : STD_LOGIC;
  signal \alu_out[18]_INST_0_i_6_n_0\ : STD_LOGIC;
  signal \alu_out[18]_INST_0_i_7_n_0\ : STD_LOGIC;
  signal \alu_out[19]_INST_0_i_1_n_0\ : STD_LOGIC;
  signal \alu_out[19]_INST_0_i_2_n_0\ : STD_LOGIC;
  signal \alu_out[19]_INST_0_i_4_n_0\ : STD_LOGIC;
  signal \alu_out[19]_INST_0_i_5_n_0\ : STD_LOGIC;
  signal \alu_out[19]_INST_0_i_6_n_0\ : STD_LOGIC;
  signal \alu_out[1]_INST_0_i_1_n_0\ : STD_LOGIC;
  signal \alu_out[1]_INST_0_i_2_n_0\ : STD_LOGIC;
  signal \alu_out[1]_INST_0_i_4_n_0\ : STD_LOGIC;
  signal \alu_out[1]_INST_0_i_5_n_0\ : STD_LOGIC;
  signal \alu_out[20]_INST_0_i_1_n_0\ : STD_LOGIC;
  signal \alu_out[20]_INST_0_i_2_n_0\ : STD_LOGIC;
  signal \alu_out[20]_INST_0_i_4_n_0\ : STD_LOGIC;
  signal \alu_out[20]_INST_0_i_5_n_0\ : STD_LOGIC;
  signal \alu_out[20]_INST_0_i_6_n_0\ : STD_LOGIC;
  signal \alu_out[21]_INST_0_i_1_n_0\ : STD_LOGIC;
  signal \alu_out[21]_INST_0_i_2_n_0\ : STD_LOGIC;
  signal \alu_out[21]_INST_0_i_4_n_0\ : STD_LOGIC;
  signal \alu_out[21]_INST_0_i_5_n_0\ : STD_LOGIC;
  signal \alu_out[21]_INST_0_i_6_n_0\ : STD_LOGIC;
  signal \alu_out[22]_INST_0_i_1_n_0\ : STD_LOGIC;
  signal \alu_out[22]_INST_0_i_2_n_0\ : STD_LOGIC;
  signal \alu_out[22]_INST_0_i_4_n_0\ : STD_LOGIC;
  signal \alu_out[22]_INST_0_i_5_n_0\ : STD_LOGIC;
  signal \alu_out[22]_INST_0_i_6_n_0\ : STD_LOGIC;
  signal \alu_out[23]_INST_0_i_1_n_0\ : STD_LOGIC;
  signal \alu_out[23]_INST_0_i_2_n_0\ : STD_LOGIC;
  signal \alu_out[23]_INST_0_i_4_n_0\ : STD_LOGIC;
  signal \alu_out[23]_INST_0_i_5_n_0\ : STD_LOGIC;
  signal \alu_out[23]_INST_0_i_6_n_0\ : STD_LOGIC;
  signal \alu_out[24]_INST_0_i_1_n_0\ : STD_LOGIC;
  signal \alu_out[24]_INST_0_i_2_n_0\ : STD_LOGIC;
  signal \alu_out[24]_INST_0_i_4_n_0\ : STD_LOGIC;
  signal \alu_out[24]_INST_0_i_5_n_0\ : STD_LOGIC;
  signal \alu_out[24]_INST_0_i_6_n_0\ : STD_LOGIC;
  signal \alu_out[25]_INST_0_i_1_n_0\ : STD_LOGIC;
  signal \alu_out[25]_INST_0_i_2_n_0\ : STD_LOGIC;
  signal \alu_out[25]_INST_0_i_4_n_0\ : STD_LOGIC;
  signal \alu_out[25]_INST_0_i_5_n_0\ : STD_LOGIC;
  signal \alu_out[25]_INST_0_i_6_n_0\ : STD_LOGIC;
  signal \alu_out[26]_INST_0_i_1_n_0\ : STD_LOGIC;
  signal \alu_out[26]_INST_0_i_2_n_0\ : STD_LOGIC;
  signal \alu_out[26]_INST_0_i_4_n_0\ : STD_LOGIC;
  signal \alu_out[26]_INST_0_i_5_n_0\ : STD_LOGIC;
  signal \alu_out[26]_INST_0_i_6_n_0\ : STD_LOGIC;
  signal \alu_out[27]_INST_0_i_1_n_0\ : STD_LOGIC;
  signal \alu_out[27]_INST_0_i_2_n_0\ : STD_LOGIC;
  signal \alu_out[27]_INST_0_i_4_n_0\ : STD_LOGIC;
  signal \alu_out[27]_INST_0_i_5_n_0\ : STD_LOGIC;
  signal \alu_out[27]_INST_0_i_6_n_0\ : STD_LOGIC;
  signal \alu_out[27]_INST_0_i_7_n_0\ : STD_LOGIC;
  signal \alu_out[28]_INST_0_i_1_n_0\ : STD_LOGIC;
  signal \alu_out[28]_INST_0_i_2_n_0\ : STD_LOGIC;
  signal \alu_out[28]_INST_0_i_4_n_0\ : STD_LOGIC;
  signal \alu_out[28]_INST_0_i_5_n_0\ : STD_LOGIC;
  signal \alu_out[28]_INST_0_i_6_n_0\ : STD_LOGIC;
  signal \alu_out[28]_INST_0_i_7_n_0\ : STD_LOGIC;
  signal \alu_out[29]_INST_0_i_1_n_0\ : STD_LOGIC;
  signal \alu_out[29]_INST_0_i_2_n_0\ : STD_LOGIC;
  signal \alu_out[29]_INST_0_i_4_n_0\ : STD_LOGIC;
  signal \alu_out[29]_INST_0_i_5_n_0\ : STD_LOGIC;
  signal \alu_out[2]_INST_0_i_1_n_0\ : STD_LOGIC;
  signal \alu_out[2]_INST_0_i_2_n_0\ : STD_LOGIC;
  signal \alu_out[2]_INST_0_i_4_n_0\ : STD_LOGIC;
  signal \alu_out[2]_INST_0_i_5_n_0\ : STD_LOGIC;
  signal \alu_out[30]_INST_0_i_1_n_0\ : STD_LOGIC;
  signal \alu_out[30]_INST_0_i_2_n_0\ : STD_LOGIC;
  signal \alu_out[30]_INST_0_i_4_n_0\ : STD_LOGIC;
  signal \alu_out[30]_INST_0_i_5_n_0\ : STD_LOGIC;
  signal \alu_out[31]_INST_0_i_10_n_0\ : STD_LOGIC;
  signal \alu_out[31]_INST_0_i_11_n_0\ : STD_LOGIC;
  signal \alu_out[31]_INST_0_i_12_n_0\ : STD_LOGIC;
  signal \alu_out[31]_INST_0_i_13_n_0\ : STD_LOGIC;
  signal \alu_out[31]_INST_0_i_14_n_0\ : STD_LOGIC;
  signal \alu_out[31]_INST_0_i_15_n_0\ : STD_LOGIC;
  signal \alu_out[31]_INST_0_i_1_n_0\ : STD_LOGIC;
  signal \alu_out[31]_INST_0_i_2_n_0\ : STD_LOGIC;
  signal \alu_out[31]_INST_0_i_3_n_0\ : STD_LOGIC;
  signal \alu_out[31]_INST_0_i_4_n_0\ : STD_LOGIC;
  signal \alu_out[31]_INST_0_i_6_n_0\ : STD_LOGIC;
  signal \alu_out[31]_INST_0_i_7_n_0\ : STD_LOGIC;
  signal \alu_out[31]_INST_0_i_8_n_0\ : STD_LOGIC;
  signal \alu_out[31]_INST_0_i_9_n_0\ : STD_LOGIC;
  signal \alu_out[3]_INST_0_i_1_n_0\ : STD_LOGIC;
  signal \alu_out[3]_INST_0_i_2_n_0\ : STD_LOGIC;
  signal \alu_out[3]_INST_0_i_4_n_0\ : STD_LOGIC;
  signal \alu_out[3]_INST_0_i_5_n_0\ : STD_LOGIC;
  signal \alu_out[4]_INST_0_i_1_n_0\ : STD_LOGIC;
  signal \alu_out[4]_INST_0_i_2_n_0\ : STD_LOGIC;
  signal \alu_out[4]_INST_0_i_4_n_0\ : STD_LOGIC;
  signal \alu_out[4]_INST_0_i_5_n_0\ : STD_LOGIC;
  signal \alu_out[5]_INST_0_i_1_n_0\ : STD_LOGIC;
  signal \alu_out[5]_INST_0_i_2_n_0\ : STD_LOGIC;
  signal \alu_out[5]_INST_0_i_4_n_0\ : STD_LOGIC;
  signal \alu_out[5]_INST_0_i_5_n_0\ : STD_LOGIC;
  signal \alu_out[6]_INST_0_i_1_n_0\ : STD_LOGIC;
  signal \alu_out[6]_INST_0_i_2_n_0\ : STD_LOGIC;
  signal \alu_out[6]_INST_0_i_4_n_0\ : STD_LOGIC;
  signal \alu_out[6]_INST_0_i_5_n_0\ : STD_LOGIC;
  signal \alu_out[7]_INST_0_i_1_n_0\ : STD_LOGIC;
  signal \alu_out[7]_INST_0_i_2_n_0\ : STD_LOGIC;
  signal \alu_out[7]_INST_0_i_4_n_0\ : STD_LOGIC;
  signal \alu_out[7]_INST_0_i_5_n_0\ : STD_LOGIC;
  signal \alu_out[7]_INST_0_i_6_n_0\ : STD_LOGIC;
  signal \alu_out[8]_INST_0_i_1_n_0\ : STD_LOGIC;
  signal \alu_out[8]_INST_0_i_2_n_0\ : STD_LOGIC;
  signal \alu_out[8]_INST_0_i_4_n_0\ : STD_LOGIC;
  signal \alu_out[8]_INST_0_i_5_n_0\ : STD_LOGIC;
  signal \alu_out[8]_INST_0_i_6_n_0\ : STD_LOGIC;
  signal \alu_out[9]_INST_0_i_1_n_0\ : STD_LOGIC;
  signal \alu_out[9]_INST_0_i_2_n_0\ : STD_LOGIC;
  signal \alu_out[9]_INST_0_i_4_n_0\ : STD_LOGIC;
  signal \alu_out[9]_INST_0_i_5_n_0\ : STD_LOGIC;
  signal \alu_out[9]_INST_0_i_6_n_0\ : STD_LOGIC;
  attribute SOFT_HLUTNM : string;
  attribute SOFT_HLUTNM of \alu_out[10]_INST_0_i_4\ : label is "soft_lutpair10";
  attribute SOFT_HLUTNM of \alu_out[11]_INST_0_i_4\ : label is "soft_lutpair16";
  attribute SOFT_HLUTNM of \alu_out[12]_INST_0_i_4\ : label is "soft_lutpair10";
  attribute SOFT_HLUTNM of \alu_out[15]_INST_0_i_7\ : label is "soft_lutpair6";
  attribute SOFT_HLUTNM of \alu_out[16]_INST_0_i_7\ : label is "soft_lutpair4";
  attribute SOFT_HLUTNM of \alu_out[17]_INST_0_i_1\ : label is "soft_lutpair0";
  attribute SOFT_HLUTNM of \alu_out[17]_INST_0_i_7\ : label is "soft_lutpair9";
  attribute SOFT_HLUTNM of \alu_out[18]_INST_0_i_1\ : label is "soft_lutpair1";
  attribute SOFT_HLUTNM of \alu_out[18]_INST_0_i_7\ : label is "soft_lutpair2";
  attribute SOFT_HLUTNM of \alu_out[19]_INST_0_i_2\ : label is "soft_lutpair13";
  attribute SOFT_HLUTNM of \alu_out[19]_INST_0_i_6\ : label is "soft_lutpair7";
  attribute SOFT_HLUTNM of \alu_out[20]_INST_0_i_2\ : label is "soft_lutpair11";
  attribute SOFT_HLUTNM of \alu_out[20]_INST_0_i_6\ : label is "soft_lutpair5";
  attribute SOFT_HLUTNM of \alu_out[21]_INST_0_i_2\ : label is "soft_lutpair14";
  attribute SOFT_HLUTNM of \alu_out[21]_INST_0_i_6\ : label is "soft_lutpair8";
  attribute SOFT_HLUTNM of \alu_out[22]_INST_0_i_2\ : label is "soft_lutpair11";
  attribute SOFT_HLUTNM of \alu_out[22]_INST_0_i_6\ : label is "soft_lutpair3";
  attribute SOFT_HLUTNM of \alu_out[23]_INST_0_i_2\ : label is "soft_lutpair14";
  attribute SOFT_HLUTNM of \alu_out[23]_INST_0_i_6\ : label is "soft_lutpair6";
  attribute SOFT_HLUTNM of \alu_out[24]_INST_0_i_2\ : label is "soft_lutpair12";
  attribute SOFT_HLUTNM of \alu_out[24]_INST_0_i_6\ : label is "soft_lutpair4";
  attribute SOFT_HLUTNM of \alu_out[25]_INST_0_i_2\ : label is "soft_lutpair15";
  attribute SOFT_HLUTNM of \alu_out[25]_INST_0_i_6\ : label is "soft_lutpair9";
  attribute SOFT_HLUTNM of \alu_out[26]_INST_0_i_2\ : label is "soft_lutpair12";
  attribute SOFT_HLUTNM of \alu_out[26]_INST_0_i_6\ : label is "soft_lutpair2";
  attribute SOFT_HLUTNM of \alu_out[27]_INST_0_i_2\ : label is "soft_lutpair15";
  attribute SOFT_HLUTNM of \alu_out[27]_INST_0_i_7\ : label is "soft_lutpair7";
  attribute SOFT_HLUTNM of \alu_out[28]_INST_0_i_2\ : label is "soft_lutpair13";
  attribute SOFT_HLUTNM of \alu_out[28]_INST_0_i_7\ : label is "soft_lutpair5";
  attribute SOFT_HLUTNM of \alu_out[29]_INST_0_i_5\ : label is "soft_lutpair8";
  attribute SOFT_HLUTNM of \alu_out[30]_INST_0_i_5\ : label is "soft_lutpair3";
  attribute SOFT_HLUTNM of \alu_out[31]_INST_0_i_4\ : label is "soft_lutpair1";
  attribute SOFT_HLUTNM of \alu_out[31]_INST_0_i_6\ : label is "soft_lutpair0";
  attribute SOFT_HLUTNM of \alu_out[7]_INST_0_i_4\ : label is "soft_lutpair17";
  attribute SOFT_HLUTNM of \alu_out[8]_INST_0_i_4\ : label is "soft_lutpair17";
  attribute SOFT_HLUTNM of \alu_out[9]_INST_0_i_4\ : label is "soft_lutpair16";
begin
\alu_out[0]_INST_0_i_1\: unisim.vcomponents.LUT5
    generic map(
      INIT => X"A0C0A0CF"
    )
        port map (
      I0 => \alu_out[1]_INST_0_i_2_n_0\,
      I1 => \alu_out[0]_INST_0_i_4_n_0\,
      I2 => alu_sel(0),
      I3 => alu_b(0),
      I4 => \alu_out[1]_INST_0_i_4_n_0\,
      O => \alu_out[0]_INST_0_i_1_n_0\
    );
\alu_out[0]_INST_0_i_4\: unisim.vcomponents.LUT6
    generic map(
      INIT => X"AFA0CFCFAFA0C0C0"
    )
        port map (
      I0 => \alu_out[6]_INST_0_i_5_n_0\,
      I1 => \alu_out[2]_INST_0_i_5_n_0\,
      I2 => alu_b(1),
      I3 => \alu_out[4]_INST_0_i_5_n_0\,
      I4 => alu_b(2),
      I5 => \alu_out[0]_INST_0_i_5_n_0\,
      O => \alu_out[0]_INST_0_i_4_n_0\
    );
\alu_out[0]_INST_0_i_5\: unisim.vcomponents.LUT6
    generic map(
      INIT => X"AFA0CFCFAFA0C0C0"
    )
        port map (
      I0 => alu_a(24),
      I1 => alu_a(8),
      I2 => alu_b(3),
      I3 => alu_a(16),
      I4 => alu_b(4),
      I5 => alu_a(0),
      O => \alu_out[0]_INST_0_i_5_n_0\
    );
\alu_out[10]_INST_0_i_1\: unisim.vcomponents.LUT5
    generic map(
      INIT => X"33373F37"
    )
        port map (
      I0 => \alu_out[11]_INST_0_i_4_n_0\,
      I1 => alu_sel(1),
      I2 => alu_sel(0),
      I3 => alu_b(0),
      I4 => \alu_out[10]_INST_0_i_4_n_0\,
      O => \alu_out[10]_INST_0_i_1_n_0\
    );
\alu_out[10]_INST_0_i_2\: unisim.vcomponents.LUT6
    generic map(
      INIT => X"AFA0CFCFAFA0C0C0"
    )
        port map (
      I0 => \alu_out[16]_INST_0_i_6_n_0\,
      I1 => \alu_out[12]_INST_0_i_5_n_0\,
      I2 => alu_b(1),
      I3 => \alu_out[14]_INST_0_i_5_n_0\,
      I4 => alu_b(2),
      I5 => \alu_out[10]_INST_0_i_5_n_0\,
      O => \alu_out[10]_INST_0_i_2_n_0\
    );
\alu_out[10]_INST_0_i_4\: unisim.vcomponents.LUT3
    generic map(
      INIT => X"B8"
    )
        port map (
      I0 => \alu_out[10]_INST_0_i_6_n_0\,
      I1 => alu_b(1),
      I2 => \alu_out[12]_INST_0_i_6_n_0\,
      O => \alu_out[10]_INST_0_i_4_n_0\
    );
\alu_out[10]_INST_0_i_5\: unisim.vcomponents.LUT6
    generic map(
      INIT => X"AFA0CFCFAFA0C0C0"
    )
        port map (
      I0 => alu_a(31),
      I1 => alu_a(18),
      I2 => alu_b(3),
      I3 => alu_a(26),
      I4 => alu_b(4),
      I5 => alu_a(10),
      O => \alu_out[10]_INST_0_i_5_n_0\
    );
\alu_out[10]_INST_0_i_6\: unisim.vcomponents.LUT5
    generic map(
      INIT => X"FFF4FFF7"
    )
        port map (
      I0 => alu_a(3),
      I1 => alu_b(2),
      I2 => alu_b(3),
      I3 => alu_b(4),
      I4 => alu_a(7),
      O => \alu_out[10]_INST_0_i_6_n_0\
    );
\alu_out[11]_INST_0_i_1\: unisim.vcomponents.LUT5
    generic map(
      INIT => X"33373F37"
    )
        port map (
      I0 => \alu_out[12]_INST_0_i_4_n_0\,
      I1 => alu_sel(1),
      I2 => alu_sel(0),
      I3 => alu_b(0),
      I4 => \alu_out[11]_INST_0_i_4_n_0\,
      O => \alu_out[11]_INST_0_i_1_n_0\
    );
\alu_out[11]_INST_0_i_2\: unisim.vcomponents.LUT6
    generic map(
      INIT => X"AFA0CFCFAFA0C0C0"
    )
        port map (
      I0 => \alu_out[17]_INST_0_i_6_n_0\,
      I1 => \alu_out[13]_INST_0_i_5_n_0\,
      I2 => alu_b(1),
      I3 => \alu_out[15]_INST_0_i_6_n_0\,
      I4 => alu_b(2),
      I5 => \alu_out[11]_INST_0_i_5_n_0\,
      O => \alu_out[11]_INST_0_i_2_n_0\
    );
\alu_out[11]_INST_0_i_4\: unisim.vcomponents.LUT3
    generic map(
      INIT => X"B8"
    )
        port map (
      I0 => \alu_out[11]_INST_0_i_6_n_0\,
      I1 => alu_b(1),
      I2 => \alu_out[13]_INST_0_i_6_n_0\,
      O => \alu_out[11]_INST_0_i_4_n_0\
    );
\alu_out[11]_INST_0_i_5\: unisim.vcomponents.LUT6
    generic map(
      INIT => X"AFA0CFCFAFA0C0C0"
    )
        port map (
      I0 => alu_a(31),
      I1 => alu_a(19),
      I2 => alu_b(3),
      I3 => alu_a(27),
      I4 => alu_b(4),
      I5 => alu_a(11),
      O => \alu_out[11]_INST_0_i_5_n_0\
    );
\alu_out[11]_INST_0_i_6\: unisim.vcomponents.LUT6
    generic map(
      INIT => X"FFFFFFFFCF44CF77"
    )
        port map (
      I0 => alu_a(4),
      I1 => alu_b(2),
      I2 => alu_a(0),
      I3 => alu_b(3),
      I4 => alu_a(8),
      I5 => alu_b(4),
      O => \alu_out[11]_INST_0_i_6_n_0\
    );
\alu_out[12]_INST_0_i_1\: unisim.vcomponents.LUT5
    generic map(
      INIT => X"33373F37"
    )
        port map (
      I0 => \alu_out[13]_INST_0_i_4_n_0\,
      I1 => alu_sel(1),
      I2 => alu_sel(0),
      I3 => alu_b(0),
      I4 => \alu_out[12]_INST_0_i_4_n_0\,
      O => \alu_out[12]_INST_0_i_1_n_0\
    );
\alu_out[12]_INST_0_i_2\: unisim.vcomponents.LUT6
    generic map(
      INIT => X"AFA0CFCFAFA0C0C0"
    )
        port map (
      I0 => \alu_out[18]_INST_0_i_6_n_0\,
      I1 => \alu_out[14]_INST_0_i_5_n_0\,
      I2 => alu_b(1),
      I3 => \alu_out[16]_INST_0_i_6_n_0\,
      I4 => alu_b(2),
      I5 => \alu_out[12]_INST_0_i_5_n_0\,
      O => \alu_out[12]_INST_0_i_2_n_0\
    );
\alu_out[12]_INST_0_i_4\: unisim.vcomponents.LUT3
    generic map(
      INIT => X"B8"
    )
        port map (
      I0 => \alu_out[12]_INST_0_i_6_n_0\,
      I1 => alu_b(1),
      I2 => \alu_out[14]_INST_0_i_6_n_0\,
      O => \alu_out[12]_INST_0_i_4_n_0\
    );
\alu_out[12]_INST_0_i_5\: unisim.vcomponents.LUT6
    generic map(
      INIT => X"AFA0CFCFAFA0C0C0"
    )
        port map (
      I0 => alu_a(31),
      I1 => alu_a(20),
      I2 => alu_b(3),
      I3 => alu_a(28),
      I4 => alu_b(4),
      I5 => alu_a(12),
      O => \alu_out[12]_INST_0_i_5_n_0\
    );
\alu_out[12]_INST_0_i_6\: unisim.vcomponents.LUT6
    generic map(
      INIT => X"FFFFFFFFCF44CF77"
    )
        port map (
      I0 => alu_a(5),
      I1 => alu_b(2),
      I2 => alu_a(1),
      I3 => alu_b(3),
      I4 => alu_a(9),
      I5 => alu_b(4),
      O => \alu_out[12]_INST_0_i_6_n_0\
    );
\alu_out[13]_INST_0_i_1\: unisim.vcomponents.LUT5
    generic map(
      INIT => X"33373F37"
    )
        port map (
      I0 => \alu_out[14]_INST_0_i_4_n_0\,
      I1 => alu_sel(1),
      I2 => alu_sel(0),
      I3 => alu_b(0),
      I4 => \alu_out[13]_INST_0_i_4_n_0\,
      O => \alu_out[13]_INST_0_i_1_n_0\
    );
\alu_out[13]_INST_0_i_2\: unisim.vcomponents.LUT6
    generic map(
      INIT => X"AFA0CFCFAFA0C0C0"
    )
        port map (
      I0 => \alu_out[15]_INST_0_i_5_n_0\,
      I1 => \alu_out[15]_INST_0_i_6_n_0\,
      I2 => alu_b(1),
      I3 => \alu_out[17]_INST_0_i_6_n_0\,
      I4 => alu_b(2),
      I5 => \alu_out[13]_INST_0_i_5_n_0\,
      O => \alu_out[13]_INST_0_i_2_n_0\
    );
\alu_out[13]_INST_0_i_4\: unisim.vcomponents.LUT5
    generic map(
      INIT => X"B8BBB888"
    )
        port map (
      I0 => \alu_out[13]_INST_0_i_6_n_0\,
      I1 => alu_b(1),
      I2 => \alu_out[15]_INST_0_i_7_n_0\,
      I3 => alu_b(2),
      I4 => \alu_out[19]_INST_0_i_6_n_0\,
      O => \alu_out[13]_INST_0_i_4_n_0\
    );
\alu_out[13]_INST_0_i_5\: unisim.vcomponents.LUT6
    generic map(
      INIT => X"AFA0CFCFAFA0C0C0"
    )
        port map (
      I0 => alu_a(31),
      I1 => alu_a(21),
      I2 => alu_b(3),
      I3 => alu_a(29),
      I4 => alu_b(4),
      I5 => alu_a(13),
      O => \alu_out[13]_INST_0_i_5_n_0\
    );
\alu_out[13]_INST_0_i_6\: unisim.vcomponents.LUT6
    generic map(
      INIT => X"FFFFFFFFCF44CF77"
    )
        port map (
      I0 => alu_a(6),
      I1 => alu_b(2),
      I2 => alu_a(2),
      I3 => alu_b(3),
      I4 => alu_a(10),
      I5 => alu_b(4),
      O => \alu_out[13]_INST_0_i_6_n_0\
    );
\alu_out[14]_INST_0_i_1\: unisim.vcomponents.LUT5
    generic map(
      INIT => X"33373F37"
    )
        port map (
      I0 => \alu_out[15]_INST_0_i_4_n_0\,
      I1 => alu_sel(1),
      I2 => alu_sel(0),
      I3 => alu_b(0),
      I4 => \alu_out[14]_INST_0_i_4_n_0\,
      O => \alu_out[14]_INST_0_i_1_n_0\
    );
\alu_out[14]_INST_0_i_2\: unisim.vcomponents.LUT6
    generic map(
      INIT => X"AFA0CFCFAFA0C0C0"
    )
        port map (
      I0 => \alu_out[16]_INST_0_i_5_n_0\,
      I1 => \alu_out[16]_INST_0_i_6_n_0\,
      I2 => alu_b(1),
      I3 => \alu_out[18]_INST_0_i_6_n_0\,
      I4 => alu_b(2),
      I5 => \alu_out[14]_INST_0_i_5_n_0\,
      O => \alu_out[14]_INST_0_i_2_n_0\
    );
\alu_out[14]_INST_0_i_4\: unisim.vcomponents.LUT5
    generic map(
      INIT => X"B8BBB888"
    )
        port map (
      I0 => \alu_out[14]_INST_0_i_6_n_0\,
      I1 => alu_b(1),
      I2 => \alu_out[16]_INST_0_i_7_n_0\,
      I3 => alu_b(2),
      I4 => \alu_out[20]_INST_0_i_6_n_0\,
      O => \alu_out[14]_INST_0_i_4_n_0\
    );
\alu_out[14]_INST_0_i_5\: unisim.vcomponents.LUT6
    generic map(
      INIT => X"AFA0CFCFAFA0C0C0"
    )
        port map (
      I0 => alu_a(31),
      I1 => alu_a(22),
      I2 => alu_b(3),
      I3 => alu_a(30),
      I4 => alu_b(4),
      I5 => alu_a(14),
      O => \alu_out[14]_INST_0_i_5_n_0\
    );
\alu_out[14]_INST_0_i_6\: unisim.vcomponents.LUT6
    generic map(
      INIT => X"FFFFFFFFCF44CF77"
    )
        port map (
      I0 => alu_a(7),
      I1 => alu_b(2),
      I2 => alu_a(3),
      I3 => alu_b(3),
      I4 => alu_a(11),
      I5 => alu_b(4),
      O => \alu_out[14]_INST_0_i_6_n_0\
    );
\alu_out[15]_INST_0_i_1\: unisim.vcomponents.LUT5
    generic map(
      INIT => X"33373F37"
    )
        port map (
      I0 => \alu_out[16]_INST_0_i_4_n_0\,
      I1 => alu_sel(1),
      I2 => alu_sel(0),
      I3 => alu_b(0),
      I4 => \alu_out[15]_INST_0_i_4_n_0\,
      O => \alu_out[15]_INST_0_i_1_n_0\
    );
\alu_out[15]_INST_0_i_2\: unisim.vcomponents.LUT6
    generic map(
      INIT => X"AFA0CFCFAFA0C0C0"
    )
        port map (
      I0 => \alu_out[17]_INST_0_i_5_n_0\,
      I1 => \alu_out[17]_INST_0_i_6_n_0\,
      I2 => alu_b(1),
      I3 => \alu_out[15]_INST_0_i_5_n_0\,
      I4 => alu_b(2),
      I5 => \alu_out[15]_INST_0_i_6_n_0\,
      O => \alu_out[15]_INST_0_i_2_n_0\
    );
\alu_out[15]_INST_0_i_4\: unisim.vcomponents.LUT6
    generic map(
      INIT => X"AFAFC0CFA0A0C0CF"
    )
        port map (
      I0 => \alu_out[15]_INST_0_i_7_n_0\,
      I1 => \alu_out[19]_INST_0_i_6_n_0\,
      I2 => alu_b(1),
      I3 => \alu_out[21]_INST_0_i_6_n_0\,
      I4 => alu_b(2),
      I5 => \alu_out[17]_INST_0_i_7_n_0\,
      O => \alu_out[15]_INST_0_i_4_n_0\
    );
\alu_out[15]_INST_0_i_5\: unisim.vcomponents.LUT5
    generic map(
      INIT => X"F0BBF088"
    )
        port map (
      I0 => alu_a(27),
      I1 => alu_b(3),
      I2 => alu_a(31),
      I3 => alu_b(4),
      I4 => alu_a(19),
      O => \alu_out[15]_INST_0_i_5_n_0\
    );
\alu_out[15]_INST_0_i_6\: unisim.vcomponents.LUT5
    generic map(
      INIT => X"F0BBF088"
    )
        port map (
      I0 => alu_a(23),
      I1 => alu_b(3),
      I2 => alu_a(31),
      I3 => alu_b(4),
      I4 => alu_a(15),
      O => \alu_out[15]_INST_0_i_6_n_0\
    );
\alu_out[15]_INST_0_i_7\: unisim.vcomponents.LUT4
    generic map(
      INIT => X"FF47"
    )
        port map (
      I0 => alu_a(0),
      I1 => alu_b(3),
      I2 => alu_a(8),
      I3 => alu_b(4),
      O => \alu_out[15]_INST_0_i_7_n_0\
    );
\alu_out[16]_INST_0_i_1\: unisim.vcomponents.LUT5
    generic map(
      INIT => X"333B3F3B"
    )
        port map (
      I0 => \alu_out[17]_INST_0_i_4_n_0\,
      I1 => alu_sel(1),
      I2 => alu_sel(0),
      I3 => alu_b(0),
      I4 => \alu_out[16]_INST_0_i_4_n_0\,
      O => \alu_out[16]_INST_0_i_1_n_0\
    );
\alu_out[16]_INST_0_i_2\: unisim.vcomponents.LUT6
    generic map(
      INIT => X"AFA0CFCFAFA0C0C0"
    )
        port map (
      I0 => \alu_out[18]_INST_0_i_5_n_0\,
      I1 => \alu_out[18]_INST_0_i_6_n_0\,
      I2 => alu_b(1),
      I3 => \alu_out[16]_INST_0_i_5_n_0\,
      I4 => alu_b(2),
      I5 => \alu_out[16]_INST_0_i_6_n_0\,
      O => \alu_out[16]_INST_0_i_2_n_0\
    );
\alu_out[16]_INST_0_i_4\: unisim.vcomponents.LUT6
    generic map(
      INIT => X"A0AFC0C0A0AFCFCF"
    )
        port map (
      I0 => \alu_out[16]_INST_0_i_7_n_0\,
      I1 => \alu_out[20]_INST_0_i_6_n_0\,
      I2 => alu_b(1),
      I3 => \alu_out[18]_INST_0_i_7_n_0\,
      I4 => alu_b(2),
      I5 => \alu_out[22]_INST_0_i_6_n_0\,
      O => \alu_out[16]_INST_0_i_4_n_0\
    );
\alu_out[16]_INST_0_i_5\: unisim.vcomponents.LUT5
    generic map(
      INIT => X"F0BBF088"
    )
        port map (
      I0 => alu_a(28),
      I1 => alu_b(3),
      I2 => alu_a(31),
      I3 => alu_b(4),
      I4 => alu_a(20),
      O => \alu_out[16]_INST_0_i_5_n_0\
    );
\alu_out[16]_INST_0_i_6\: unisim.vcomponents.LUT5
    generic map(
      INIT => X"F0BBF088"
    )
        port map (
      I0 => alu_a(24),
      I1 => alu_b(3),
      I2 => alu_a(31),
      I3 => alu_b(4),
      I4 => alu_a(16),
      O => \alu_out[16]_INST_0_i_6_n_0\
    );
\alu_out[16]_INST_0_i_7\: unisim.vcomponents.LUT4
    generic map(
      INIT => X"FF47"
    )
        port map (
      I0 => alu_a(1),
      I1 => alu_b(3),
      I2 => alu_a(9),
      I3 => alu_b(4),
      O => \alu_out[16]_INST_0_i_7_n_0\
    );
\alu_out[17]_INST_0_i_1\: unisim.vcomponents.LUT5
    generic map(
      INIT => X"33F333BB"
    )
        port map (
      I0 => \alu_out[18]_INST_0_i_4_n_0\,
      I1 => alu_sel(1),
      I2 => \alu_out[17]_INST_0_i_4_n_0\,
      I3 => alu_sel(0),
      I4 => alu_b(0),
      O => \alu_out[17]_INST_0_i_1_n_0\
    );
\alu_out[17]_INST_0_i_2\: unisim.vcomponents.LUT5
    generic map(
      INIT => X"FF00B8B8"
    )
        port map (
      I0 => \alu_out[17]_INST_0_i_5_n_0\,
      I1 => alu_b(2),
      I2 => \alu_out[17]_INST_0_i_6_n_0\,
      I3 => \alu_out[19]_INST_0_i_5_n_0\,
      I4 => alu_b(1),
      O => \alu_out[17]_INST_0_i_2_n_0\
    );
\alu_out[17]_INST_0_i_4\: unisim.vcomponents.LUT6
    generic map(
      INIT => X"505FCFCF505FC0C0"
    )
        port map (
      I0 => \alu_out[17]_INST_0_i_7_n_0\,
      I1 => \alu_out[21]_INST_0_i_6_n_0\,
      I2 => alu_b(1),
      I3 => \alu_out[19]_INST_0_i_6_n_0\,
      I4 => alu_b(2),
      I5 => \alu_out[23]_INST_0_i_6_n_0\,
      O => \alu_out[17]_INST_0_i_4_n_0\
    );
\alu_out[17]_INST_0_i_5\: unisim.vcomponents.LUT5
    generic map(
      INIT => X"F0BBF088"
    )
        port map (
      I0 => alu_a(29),
      I1 => alu_b(3),
      I2 => alu_a(31),
      I3 => alu_b(4),
      I4 => alu_a(21),
      O => \alu_out[17]_INST_0_i_5_n_0\
    );
\alu_out[17]_INST_0_i_6\: unisim.vcomponents.LUT5
    generic map(
      INIT => X"F0BBF088"
    )
        port map (
      I0 => alu_a(25),
      I1 => alu_b(3),
      I2 => alu_a(31),
      I3 => alu_b(4),
      I4 => alu_a(17),
      O => \alu_out[17]_INST_0_i_6_n_0\
    );
\alu_out[17]_INST_0_i_7\: unisim.vcomponents.LUT4
    generic map(
      INIT => X"FF47"
    )
        port map (
      I0 => alu_a(2),
      I1 => alu_b(3),
      I2 => alu_a(10),
      I3 => alu_b(4),
      O => \alu_out[17]_INST_0_i_7_n_0\
    );
\alu_out[18]_INST_0_i_1\: unisim.vcomponents.LUT5
    generic map(
      INIT => X"33F333BB"
    )
        port map (
      I0 => \alu_out[19]_INST_0_i_4_n_0\,
      I1 => alu_sel(1),
      I2 => \alu_out[18]_INST_0_i_4_n_0\,
      I3 => alu_sel(0),
      I4 => alu_b(0),
      O => \alu_out[18]_INST_0_i_1_n_0\
    );
\alu_out[18]_INST_0_i_2\: unisim.vcomponents.LUT5
    generic map(
      INIT => X"FF00B8B8"
    )
        port map (
      I0 => \alu_out[18]_INST_0_i_5_n_0\,
      I1 => alu_b(2),
      I2 => \alu_out[18]_INST_0_i_6_n_0\,
      I3 => \alu_out[20]_INST_0_i_5_n_0\,
      I4 => alu_b(1),
      O => \alu_out[18]_INST_0_i_2_n_0\
    );
\alu_out[18]_INST_0_i_4\: unisim.vcomponents.LUT6
    generic map(
      INIT => X"A0AFCFCFA0AFC0C0"
    )
        port map (
      I0 => \alu_out[18]_INST_0_i_7_n_0\,
      I1 => \alu_out[22]_INST_0_i_6_n_0\,
      I2 => alu_b(1),
      I3 => \alu_out[20]_INST_0_i_6_n_0\,
      I4 => alu_b(2),
      I5 => \alu_out[24]_INST_0_i_6_n_0\,
      O => \alu_out[18]_INST_0_i_4_n_0\
    );
\alu_out[18]_INST_0_i_5\: unisim.vcomponents.LUT5
    generic map(
      INIT => X"F0BBF088"
    )
        port map (
      I0 => alu_a(30),
      I1 => alu_b(3),
      I2 => alu_a(31),
      I3 => alu_b(4),
      I4 => alu_a(22),
      O => \alu_out[18]_INST_0_i_5_n_0\
    );
\alu_out[18]_INST_0_i_6\: unisim.vcomponents.LUT5
    generic map(
      INIT => X"F0BBF088"
    )
        port map (
      I0 => alu_a(26),
      I1 => alu_b(3),
      I2 => alu_a(31),
      I3 => alu_b(4),
      I4 => alu_a(18),
      O => \alu_out[18]_INST_0_i_6_n_0\
    );
\alu_out[18]_INST_0_i_7\: unisim.vcomponents.LUT4
    generic map(
      INIT => X"00B8"
    )
        port map (
      I0 => alu_a(3),
      I1 => alu_b(3),
      I2 => alu_a(11),
      I3 => alu_b(4),
      O => \alu_out[18]_INST_0_i_7_n_0\
    );
\alu_out[19]_INST_0_i_1\: unisim.vcomponents.LUT5
    generic map(
      INIT => X"33F333BB"
    )
        port map (
      I0 => \alu_out[20]_INST_0_i_4_n_0\,
      I1 => alu_sel(1),
      I2 => \alu_out[19]_INST_0_i_4_n_0\,
      I3 => alu_sel(0),
      I4 => alu_b(0),
      O => \alu_out[19]_INST_0_i_1_n_0\
    );
\alu_out[19]_INST_0_i_2\: unisim.vcomponents.LUT3
    generic map(
      INIT => X"B8"
    )
        port map (
      I0 => \alu_out[21]_INST_0_i_5_n_0\,
      I1 => alu_b(1),
      I2 => \alu_out[19]_INST_0_i_5_n_0\,
      O => \alu_out[19]_INST_0_i_2_n_0\
    );
\alu_out[19]_INST_0_i_4\: unisim.vcomponents.LUT6
    generic map(
      INIT => X"5F50CFCF5F50C0C0"
    )
        port map (
      I0 => \alu_out[19]_INST_0_i_6_n_0\,
      I1 => \alu_out[23]_INST_0_i_6_n_0\,
      I2 => alu_b(1),
      I3 => \alu_out[21]_INST_0_i_6_n_0\,
      I4 => alu_b(2),
      I5 => \alu_out[25]_INST_0_i_6_n_0\,
      O => \alu_out[19]_INST_0_i_4_n_0\
    );
\alu_out[19]_INST_0_i_5\: unisim.vcomponents.LUT6
    generic map(
      INIT => X"CDC8FFFFCDC80000"
    )
        port map (
      I0 => alu_b(3),
      I1 => alu_a(31),
      I2 => alu_b(4),
      I3 => alu_a(23),
      I4 => alu_b(2),
      I5 => \alu_out[15]_INST_0_i_5_n_0\,
      O => \alu_out[19]_INST_0_i_5_n_0\
    );
\alu_out[19]_INST_0_i_6\: unisim.vcomponents.LUT4
    generic map(
      INIT => X"FF47"
    )
        port map (
      I0 => alu_a(4),
      I1 => alu_b(3),
      I2 => alu_a(12),
      I3 => alu_b(4),
      O => \alu_out[19]_INST_0_i_6_n_0\
    );
\alu_out[1]_INST_0_i_1\: unisim.vcomponents.LUT5
    generic map(
      INIT => X"55555F77"
    )
        port map (
      I0 => alu_sel(1),
      I1 => \alu_out[2]_INST_0_i_4_n_0\,
      I2 => \alu_out[1]_INST_0_i_4_n_0\,
      I3 => alu_b(0),
      I4 => alu_sel(0),
      O => \alu_out[1]_INST_0_i_1_n_0\
    );
\alu_out[1]_INST_0_i_2\: unisim.vcomponents.LUT6
    generic map(
      INIT => X"AFA0CFCFAFA0C0C0"
    )
        port map (
      I0 => \alu_out[7]_INST_0_i_5_n_0\,
      I1 => \alu_out[3]_INST_0_i_5_n_0\,
      I2 => alu_b(1),
      I3 => \alu_out[5]_INST_0_i_5_n_0\,
      I4 => alu_b(2),
      I5 => \alu_out[1]_INST_0_i_5_n_0\,
      O => \alu_out[1]_INST_0_i_2_n_0\
    );
\alu_out[1]_INST_0_i_4\: unisim.vcomponents.LUT5
    generic map(
      INIT => X"FFFFFFEF"
    )
        port map (
      I0 => alu_b(1),
      I1 => alu_b(3),
      I2 => alu_a(0),
      I3 => alu_b(4),
      I4 => alu_b(2),
      O => \alu_out[1]_INST_0_i_4_n_0\
    );
\alu_out[1]_INST_0_i_5\: unisim.vcomponents.LUT6
    generic map(
      INIT => X"AFA0CFCFAFA0C0C0"
    )
        port map (
      I0 => alu_a(25),
      I1 => alu_a(9),
      I2 => alu_b(3),
      I3 => alu_a(17),
      I4 => alu_b(4),
      I5 => alu_a(1),
      O => \alu_out[1]_INST_0_i_5_n_0\
    );
\alu_out[20]_INST_0_i_1\: unisim.vcomponents.LUT5
    generic map(
      INIT => X"33F333BB"
    )
        port map (
      I0 => \alu_out[21]_INST_0_i_4_n_0\,
      I1 => alu_sel(1),
      I2 => \alu_out[20]_INST_0_i_4_n_0\,
      I3 => alu_sel(0),
      I4 => alu_b(0),
      O => \alu_out[20]_INST_0_i_1_n_0\
    );
\alu_out[20]_INST_0_i_2\: unisim.vcomponents.LUT3
    generic map(
      INIT => X"B8"
    )
        port map (
      I0 => \alu_out[22]_INST_0_i_5_n_0\,
      I1 => alu_b(1),
      I2 => \alu_out[20]_INST_0_i_5_n_0\,
      O => \alu_out[20]_INST_0_i_2_n_0\
    );
\alu_out[20]_INST_0_i_4\: unisim.vcomponents.LUT6
    generic map(
      INIT => X"5F50CFCF5F50C0C0"
    )
        port map (
      I0 => \alu_out[20]_INST_0_i_6_n_0\,
      I1 => \alu_out[24]_INST_0_i_6_n_0\,
      I2 => alu_b(1),
      I3 => \alu_out[22]_INST_0_i_6_n_0\,
      I4 => alu_b(2),
      I5 => \alu_out[26]_INST_0_i_6_n_0\,
      O => \alu_out[20]_INST_0_i_4_n_0\
    );
\alu_out[20]_INST_0_i_5\: unisim.vcomponents.LUT6
    generic map(
      INIT => X"CDC8FFFFCDC80000"
    )
        port map (
      I0 => alu_b(3),
      I1 => alu_a(31),
      I2 => alu_b(4),
      I3 => alu_a(24),
      I4 => alu_b(2),
      I5 => \alu_out[16]_INST_0_i_5_n_0\,
      O => \alu_out[20]_INST_0_i_5_n_0\
    );
\alu_out[20]_INST_0_i_6\: unisim.vcomponents.LUT4
    generic map(
      INIT => X"FF47"
    )
        port map (
      I0 => alu_a(5),
      I1 => alu_b(3),
      I2 => alu_a(13),
      I3 => alu_b(4),
      O => \alu_out[20]_INST_0_i_6_n_0\
    );
\alu_out[21]_INST_0_i_1\: unisim.vcomponents.LUT5
    generic map(
      INIT => X"33F333BB"
    )
        port map (
      I0 => \alu_out[22]_INST_0_i_4_n_0\,
      I1 => alu_sel(1),
      I2 => \alu_out[21]_INST_0_i_4_n_0\,
      I3 => alu_sel(0),
      I4 => alu_b(0),
      O => \alu_out[21]_INST_0_i_1_n_0\
    );
\alu_out[21]_INST_0_i_2\: unisim.vcomponents.LUT3
    generic map(
      INIT => X"B8"
    )
        port map (
      I0 => \alu_out[23]_INST_0_i_5_n_0\,
      I1 => alu_b(1),
      I2 => \alu_out[21]_INST_0_i_5_n_0\,
      O => \alu_out[21]_INST_0_i_2_n_0\
    );
\alu_out[21]_INST_0_i_4\: unisim.vcomponents.LUT6
    generic map(
      INIT => X"AFA0CFCFAFA0C0C0"
    )
        port map (
      I0 => \alu_out[21]_INST_0_i_6_n_0\,
      I1 => \alu_out[25]_INST_0_i_6_n_0\,
      I2 => alu_b(1),
      I3 => \alu_out[23]_INST_0_i_6_n_0\,
      I4 => alu_b(2),
      I5 => \alu_out[27]_INST_0_i_7_n_0\,
      O => \alu_out[21]_INST_0_i_4_n_0\
    );
\alu_out[21]_INST_0_i_5\: unisim.vcomponents.LUT6
    generic map(
      INIT => X"CDC8FFFFCDC80000"
    )
        port map (
      I0 => alu_b(3),
      I1 => alu_a(31),
      I2 => alu_b(4),
      I3 => alu_a(25),
      I4 => alu_b(2),
      I5 => \alu_out[17]_INST_0_i_5_n_0\,
      O => \alu_out[21]_INST_0_i_5_n_0\
    );
\alu_out[21]_INST_0_i_6\: unisim.vcomponents.LUT4
    generic map(
      INIT => X"00B8"
    )
        port map (
      I0 => alu_a(6),
      I1 => alu_b(3),
      I2 => alu_a(14),
      I3 => alu_b(4),
      O => \alu_out[21]_INST_0_i_6_n_0\
    );
\alu_out[22]_INST_0_i_1\: unisim.vcomponents.LUT5
    generic map(
      INIT => X"33F333BB"
    )
        port map (
      I0 => \alu_out[23]_INST_0_i_4_n_0\,
      I1 => alu_sel(1),
      I2 => \alu_out[22]_INST_0_i_4_n_0\,
      I3 => alu_sel(0),
      I4 => alu_b(0),
      O => \alu_out[22]_INST_0_i_1_n_0\
    );
\alu_out[22]_INST_0_i_2\: unisim.vcomponents.LUT3
    generic map(
      INIT => X"B8"
    )
        port map (
      I0 => \alu_out[24]_INST_0_i_5_n_0\,
      I1 => alu_b(1),
      I2 => \alu_out[22]_INST_0_i_5_n_0\,
      O => \alu_out[22]_INST_0_i_2_n_0\
    );
\alu_out[22]_INST_0_i_4\: unisim.vcomponents.LUT6
    generic map(
      INIT => X"AFA0CFCFAFA0C0C0"
    )
        port map (
      I0 => \alu_out[22]_INST_0_i_6_n_0\,
      I1 => \alu_out[26]_INST_0_i_6_n_0\,
      I2 => alu_b(1),
      I3 => \alu_out[24]_INST_0_i_6_n_0\,
      I4 => alu_b(2),
      I5 => \alu_out[28]_INST_0_i_7_n_0\,
      O => \alu_out[22]_INST_0_i_4_n_0\
    );
\alu_out[22]_INST_0_i_5\: unisim.vcomponents.LUT6
    generic map(
      INIT => X"CDC8FFFFCDC80000"
    )
        port map (
      I0 => alu_b(3),
      I1 => alu_a(31),
      I2 => alu_b(4),
      I3 => alu_a(26),
      I4 => alu_b(2),
      I5 => \alu_out[18]_INST_0_i_5_n_0\,
      O => \alu_out[22]_INST_0_i_5_n_0\
    );
\alu_out[22]_INST_0_i_6\: unisim.vcomponents.LUT4
    generic map(
      INIT => X"00B8"
    )
        port map (
      I0 => alu_a(7),
      I1 => alu_b(3),
      I2 => alu_a(15),
      I3 => alu_b(4),
      O => \alu_out[22]_INST_0_i_6_n_0\
    );
\alu_out[23]_INST_0_i_1\: unisim.vcomponents.LUT5
    generic map(
      INIT => X"33F333BB"
    )
        port map (
      I0 => \alu_out[24]_INST_0_i_4_n_0\,
      I1 => alu_sel(1),
      I2 => \alu_out[23]_INST_0_i_4_n_0\,
      I3 => alu_sel(0),
      I4 => alu_b(0),
      O => \alu_out[23]_INST_0_i_1_n_0\
    );
\alu_out[23]_INST_0_i_2\: unisim.vcomponents.LUT3
    generic map(
      INIT => X"B8"
    )
        port map (
      I0 => \alu_out[25]_INST_0_i_5_n_0\,
      I1 => alu_b(1),
      I2 => \alu_out[23]_INST_0_i_5_n_0\,
      O => \alu_out[23]_INST_0_i_2_n_0\
    );
\alu_out[23]_INST_0_i_4\: unisim.vcomponents.LUT6
    generic map(
      INIT => X"AFA0CFCFAFA0C0C0"
    )
        port map (
      I0 => \alu_out[23]_INST_0_i_6_n_0\,
      I1 => \alu_out[27]_INST_0_i_7_n_0\,
      I2 => alu_b(1),
      I3 => \alu_out[25]_INST_0_i_6_n_0\,
      I4 => alu_b(2),
      I5 => \alu_out[29]_INST_0_i_5_n_0\,
      O => \alu_out[23]_INST_0_i_4_n_0\
    );
\alu_out[23]_INST_0_i_5\: unisim.vcomponents.LUT6
    generic map(
      INIT => X"FF00FB0BFF00F808"
    )
        port map (
      I0 => alu_a(27),
      I1 => alu_b(2),
      I2 => alu_b(3),
      I3 => alu_a(31),
      I4 => alu_b(4),
      I5 => alu_a(23),
      O => \alu_out[23]_INST_0_i_5_n_0\
    );
\alu_out[23]_INST_0_i_6\: unisim.vcomponents.LUT5
    generic map(
      INIT => X"30BB3088"
    )
        port map (
      I0 => alu_a(8),
      I1 => alu_b(3),
      I2 => alu_a(0),
      I3 => alu_b(4),
      I4 => alu_a(16),
      O => \alu_out[23]_INST_0_i_6_n_0\
    );
\alu_out[24]_INST_0_i_1\: unisim.vcomponents.LUT5
    generic map(
      INIT => X"33F333BB"
    )
        port map (
      I0 => \alu_out[25]_INST_0_i_4_n_0\,
      I1 => alu_sel(1),
      I2 => \alu_out[24]_INST_0_i_4_n_0\,
      I3 => alu_sel(0),
      I4 => alu_b(0),
      O => \alu_out[24]_INST_0_i_1_n_0\
    );
\alu_out[24]_INST_0_i_2\: unisim.vcomponents.LUT3
    generic map(
      INIT => X"B8"
    )
        port map (
      I0 => \alu_out[26]_INST_0_i_5_n_0\,
      I1 => alu_b(1),
      I2 => \alu_out[24]_INST_0_i_5_n_0\,
      O => \alu_out[24]_INST_0_i_2_n_0\
    );
\alu_out[24]_INST_0_i_4\: unisim.vcomponents.LUT6
    generic map(
      INIT => X"AFA0CFCFAFA0C0C0"
    )
        port map (
      I0 => \alu_out[24]_INST_0_i_6_n_0\,
      I1 => \alu_out[28]_INST_0_i_7_n_0\,
      I2 => alu_b(1),
      I3 => \alu_out[26]_INST_0_i_6_n_0\,
      I4 => alu_b(2),
      I5 => \alu_out[30]_INST_0_i_5_n_0\,
      O => \alu_out[24]_INST_0_i_4_n_0\
    );
\alu_out[24]_INST_0_i_5\: unisim.vcomponents.LUT6
    generic map(
      INIT => X"FF00FB0BFF00F808"
    )
        port map (
      I0 => alu_a(28),
      I1 => alu_b(2),
      I2 => alu_b(3),
      I3 => alu_a(31),
      I4 => alu_b(4),
      I5 => alu_a(24),
      O => \alu_out[24]_INST_0_i_5_n_0\
    );
\alu_out[24]_INST_0_i_6\: unisim.vcomponents.LUT5
    generic map(
      INIT => X"30BB3088"
    )
        port map (
      I0 => alu_a(9),
      I1 => alu_b(3),
      I2 => alu_a(1),
      I3 => alu_b(4),
      I4 => alu_a(17),
      O => \alu_out[24]_INST_0_i_6_n_0\
    );
\alu_out[25]_INST_0_i_1\: unisim.vcomponents.LUT5
    generic map(
      INIT => X"33F333BB"
    )
        port map (
      I0 => \alu_out[26]_INST_0_i_4_n_0\,
      I1 => alu_sel(1),
      I2 => \alu_out[25]_INST_0_i_4_n_0\,
      I3 => alu_sel(0),
      I4 => alu_b(0),
      O => \alu_out[25]_INST_0_i_1_n_0\
    );
\alu_out[25]_INST_0_i_2\: unisim.vcomponents.LUT3
    generic map(
      INIT => X"B8"
    )
        port map (
      I0 => \alu_out[27]_INST_0_i_6_n_0\,
      I1 => alu_b(1),
      I2 => \alu_out[25]_INST_0_i_5_n_0\,
      O => \alu_out[25]_INST_0_i_2_n_0\
    );
\alu_out[25]_INST_0_i_4\: unisim.vcomponents.LUT6
    generic map(
      INIT => X"AFA0CFCFAFA0C0C0"
    )
        port map (
      I0 => \alu_out[25]_INST_0_i_6_n_0\,
      I1 => \alu_out[29]_INST_0_i_5_n_0\,
      I2 => alu_b(1),
      I3 => \alu_out[27]_INST_0_i_7_n_0\,
      I4 => alu_b(2),
      I5 => \alu_out[31]_INST_0_i_10_n_0\,
      O => \alu_out[25]_INST_0_i_4_n_0\
    );
\alu_out[25]_INST_0_i_5\: unisim.vcomponents.LUT6
    generic map(
      INIT => X"FF00FB0BFF00F808"
    )
        port map (
      I0 => alu_a(29),
      I1 => alu_b(2),
      I2 => alu_b(3),
      I3 => alu_a(31),
      I4 => alu_b(4),
      I5 => alu_a(25),
      O => \alu_out[25]_INST_0_i_5_n_0\
    );
\alu_out[25]_INST_0_i_6\: unisim.vcomponents.LUT5
    generic map(
      INIT => X"30BB3088"
    )
        port map (
      I0 => alu_a(10),
      I1 => alu_b(3),
      I2 => alu_a(2),
      I3 => alu_b(4),
      I4 => alu_a(18),
      O => \alu_out[25]_INST_0_i_6_n_0\
    );
\alu_out[26]_INST_0_i_1\: unisim.vcomponents.LUT5
    generic map(
      INIT => X"33F333BB"
    )
        port map (
      I0 => \alu_out[27]_INST_0_i_4_n_0\,
      I1 => alu_sel(1),
      I2 => \alu_out[26]_INST_0_i_4_n_0\,
      I3 => alu_sel(0),
      I4 => alu_b(0),
      O => \alu_out[26]_INST_0_i_1_n_0\
    );
\alu_out[26]_INST_0_i_2\: unisim.vcomponents.LUT3
    generic map(
      INIT => X"B8"
    )
        port map (
      I0 => \alu_out[28]_INST_0_i_6_n_0\,
      I1 => alu_b(1),
      I2 => \alu_out[26]_INST_0_i_5_n_0\,
      O => \alu_out[26]_INST_0_i_2_n_0\
    );
\alu_out[26]_INST_0_i_4\: unisim.vcomponents.LUT6
    generic map(
      INIT => X"AFA0CFCFAFA0C0C0"
    )
        port map (
      I0 => \alu_out[26]_INST_0_i_6_n_0\,
      I1 => \alu_out[30]_INST_0_i_5_n_0\,
      I2 => alu_b(1),
      I3 => \alu_out[28]_INST_0_i_7_n_0\,
      I4 => alu_b(2),
      I5 => \alu_out[31]_INST_0_i_14_n_0\,
      O => \alu_out[26]_INST_0_i_4_n_0\
    );
\alu_out[26]_INST_0_i_5\: unisim.vcomponents.LUT6
    generic map(
      INIT => X"FF00FB0BFF00F808"
    )
        port map (
      I0 => alu_a(30),
      I1 => alu_b(2),
      I2 => alu_b(3),
      I3 => alu_a(31),
      I4 => alu_b(4),
      I5 => alu_a(26),
      O => \alu_out[26]_INST_0_i_5_n_0\
    );
\alu_out[26]_INST_0_i_6\: unisim.vcomponents.LUT5
    generic map(
      INIT => X"30BB3088"
    )
        port map (
      I0 => alu_a(11),
      I1 => alu_b(3),
      I2 => alu_a(3),
      I3 => alu_b(4),
      I4 => alu_a(19),
      O => \alu_out[26]_INST_0_i_6_n_0\
    );
\alu_out[27]_INST_0_i_1\: unisim.vcomponents.LUT5
    generic map(
      INIT => X"33F333BB"
    )
        port map (
      I0 => \alu_out[28]_INST_0_i_4_n_0\,
      I1 => alu_sel(1),
      I2 => \alu_out[27]_INST_0_i_4_n_0\,
      I3 => alu_sel(0),
      I4 => alu_b(0),
      O => \alu_out[27]_INST_0_i_1_n_0\
    );
\alu_out[27]_INST_0_i_2\: unisim.vcomponents.LUT3
    generic map(
      INIT => X"B8"
    )
        port map (
      I0 => \alu_out[27]_INST_0_i_5_n_0\,
      I1 => alu_b(1),
      I2 => \alu_out[27]_INST_0_i_6_n_0\,
      O => \alu_out[27]_INST_0_i_2_n_0\
    );
\alu_out[27]_INST_0_i_4\: unisim.vcomponents.LUT6
    generic map(
      INIT => X"AFA0CFCFAFA0C0C0"
    )
        port map (
      I0 => \alu_out[27]_INST_0_i_7_n_0\,
      I1 => \alu_out[31]_INST_0_i_10_n_0\,
      I2 => alu_b(1),
      I3 => \alu_out[29]_INST_0_i_5_n_0\,
      I4 => alu_b(2),
      I5 => \alu_out[31]_INST_0_i_12_n_0\,
      O => \alu_out[27]_INST_0_i_4_n_0\
    );
\alu_out[27]_INST_0_i_5\: unisim.vcomponents.LUT5
    generic map(
      INIT => X"F0F1F0E0"
    )
        port map (
      I0 => alu_b(2),
      I1 => alu_b(3),
      I2 => alu_a(31),
      I3 => alu_b(4),
      I4 => alu_a(29),
      O => \alu_out[27]_INST_0_i_5_n_0\
    );
\alu_out[27]_INST_0_i_6\: unisim.vcomponents.LUT5
    generic map(
      INIT => X"F0F1F0E0"
    )
        port map (
      I0 => alu_b(2),
      I1 => alu_b(3),
      I2 => alu_a(31),
      I3 => alu_b(4),
      I4 => alu_a(27),
      O => \alu_out[27]_INST_0_i_6_n_0\
    );
\alu_out[27]_INST_0_i_7\: unisim.vcomponents.LUT5
    generic map(
      INIT => X"3300B8B8"
    )
        port map (
      I0 => alu_a(12),
      I1 => alu_b(3),
      I2 => alu_a(20),
      I3 => alu_a(4),
      I4 => alu_b(4),
      O => \alu_out[27]_INST_0_i_7_n_0\
    );
\alu_out[28]_INST_0_i_1\: unisim.vcomponents.LUT5
    generic map(
      INIT => X"33F333BB"
    )
        port map (
      I0 => \alu_out[29]_INST_0_i_4_n_0\,
      I1 => alu_sel(1),
      I2 => \alu_out[28]_INST_0_i_4_n_0\,
      I3 => alu_sel(0),
      I4 => alu_b(0),
      O => \alu_out[28]_INST_0_i_1_n_0\
    );
\alu_out[28]_INST_0_i_2\: unisim.vcomponents.LUT3
    generic map(
      INIT => X"B8"
    )
        port map (
      I0 => \alu_out[28]_INST_0_i_5_n_0\,
      I1 => alu_b(1),
      I2 => \alu_out[28]_INST_0_i_6_n_0\,
      O => \alu_out[28]_INST_0_i_2_n_0\
    );
\alu_out[28]_INST_0_i_4\: unisim.vcomponents.LUT6
    generic map(
      INIT => X"AFA0CFCFAFA0C0C0"
    )
        port map (
      I0 => \alu_out[28]_INST_0_i_7_n_0\,
      I1 => \alu_out[31]_INST_0_i_14_n_0\,
      I2 => alu_b(1),
      I3 => \alu_out[30]_INST_0_i_5_n_0\,
      I4 => alu_b(2),
      I5 => \alu_out[31]_INST_0_i_8_n_0\,
      O => \alu_out[28]_INST_0_i_4_n_0\
    );
\alu_out[28]_INST_0_i_5\: unisim.vcomponents.LUT5
    generic map(
      INIT => X"F0F1F0E0"
    )
        port map (
      I0 => alu_b(2),
      I1 => alu_b(3),
      I2 => alu_a(31),
      I3 => alu_b(4),
      I4 => alu_a(30),
      O => \alu_out[28]_INST_0_i_5_n_0\
    );
\alu_out[28]_INST_0_i_6\: unisim.vcomponents.LUT5
    generic map(
      INIT => X"F0F1F0E0"
    )
        port map (
      I0 => alu_b(2),
      I1 => alu_b(3),
      I2 => alu_a(31),
      I3 => alu_b(4),
      I4 => alu_a(28),
      O => \alu_out[28]_INST_0_i_6_n_0\
    );
\alu_out[28]_INST_0_i_7\: unisim.vcomponents.LUT5
    generic map(
      INIT => X"30BB3088"
    )
        port map (
      I0 => alu_a(13),
      I1 => alu_b(3),
      I2 => alu_a(5),
      I3 => alu_b(4),
      I4 => alu_a(21),
      O => \alu_out[28]_INST_0_i_7_n_0\
    );
\alu_out[29]_INST_0_i_1\: unisim.vcomponents.LUT5
    generic map(
      INIT => X"33F333BB"
    )
        port map (
      I0 => \alu_out[30]_INST_0_i_4_n_0\,
      I1 => alu_sel(1),
      I2 => \alu_out[29]_INST_0_i_4_n_0\,
      I3 => alu_sel(0),
      I4 => alu_b(0),
      O => \alu_out[29]_INST_0_i_1_n_0\
    );
\alu_out[29]_INST_0_i_2\: unisim.vcomponents.LUT6
    generic map(
      INIT => X"FF00FF01FF00FE00"
    )
        port map (
      I0 => alu_b(1),
      I1 => alu_b(2),
      I2 => alu_b(3),
      I3 => alu_a(31),
      I4 => alu_b(4),
      I5 => alu_a(29),
      O => \alu_out[29]_INST_0_i_2_n_0\
    );
\alu_out[29]_INST_0_i_4\: unisim.vcomponents.LUT6
    generic map(
      INIT => X"AFA0CFCFAFA0C0C0"
    )
        port map (
      I0 => \alu_out[29]_INST_0_i_5_n_0\,
      I1 => \alu_out[31]_INST_0_i_12_n_0\,
      I2 => alu_b(1),
      I3 => \alu_out[31]_INST_0_i_10_n_0\,
      I4 => alu_b(2),
      I5 => \alu_out[31]_INST_0_i_11_n_0\,
      O => \alu_out[29]_INST_0_i_4_n_0\
    );
\alu_out[29]_INST_0_i_5\: unisim.vcomponents.LUT5
    generic map(
      INIT => X"30BB3088"
    )
        port map (
      I0 => alu_a(14),
      I1 => alu_b(3),
      I2 => alu_a(6),
      I3 => alu_b(4),
      I4 => alu_a(22),
      O => \alu_out[29]_INST_0_i_5_n_0\
    );
\alu_out[2]_INST_0_i_1\: unisim.vcomponents.LUT5
    generic map(
      INIT => X"33373F37"
    )
        port map (
      I0 => \alu_out[3]_INST_0_i_4_n_0\,
      I1 => alu_sel(1),
      I2 => alu_sel(0),
      I3 => alu_b(0),
      I4 => \alu_out[2]_INST_0_i_4_n_0\,
      O => \alu_out[2]_INST_0_i_1_n_0\
    );
\alu_out[2]_INST_0_i_2\: unisim.vcomponents.LUT6
    generic map(
      INIT => X"FF33CC00B8B8B8B8"
    )
        port map (
      I0 => \alu_out[6]_INST_0_i_5_n_0\,
      I1 => alu_b(2),
      I2 => \alu_out[2]_INST_0_i_5_n_0\,
      I3 => \alu_out[8]_INST_0_i_5_n_0\,
      I4 => \alu_out[4]_INST_0_i_5_n_0\,
      I5 => alu_b(1),
      O => \alu_out[2]_INST_0_i_2_n_0\
    );
\alu_out[2]_INST_0_i_4\: unisim.vcomponents.LUT5
    generic map(
      INIT => X"FFFFFFEF"
    )
        port map (
      I0 => alu_b(1),
      I1 => alu_b(3),
      I2 => alu_a(1),
      I3 => alu_b(4),
      I4 => alu_b(2),
      O => \alu_out[2]_INST_0_i_4_n_0\
    );
\alu_out[2]_INST_0_i_5\: unisim.vcomponents.LUT6
    generic map(
      INIT => X"AFA0CFCFAFA0C0C0"
    )
        port map (
      I0 => alu_a(26),
      I1 => alu_a(10),
      I2 => alu_b(3),
      I3 => alu_a(18),
      I4 => alu_b(4),
      I5 => alu_a(2),
      O => \alu_out[2]_INST_0_i_5_n_0\
    );
\alu_out[30]_INST_0_i_1\: unisim.vcomponents.LUT5
    generic map(
      INIT => X"33F333BB"
    )
        port map (
      I0 => \alu_out[31]_INST_0_i_3_n_0\,
      I1 => alu_sel(1),
      I2 => \alu_out[30]_INST_0_i_4_n_0\,
      I3 => alu_sel(0),
      I4 => alu_b(0),
      O => \alu_out[30]_INST_0_i_1_n_0\
    );
\alu_out[30]_INST_0_i_2\: unisim.vcomponents.LUT6
    generic map(
      INIT => X"FF00FF01FF00FE00"
    )
        port map (
      I0 => alu_b(1),
      I1 => alu_b(2),
      I2 => alu_b(3),
      I3 => alu_a(31),
      I4 => alu_b(4),
      I5 => alu_a(30),
      O => \alu_out[30]_INST_0_i_2_n_0\
    );
\alu_out[30]_INST_0_i_4\: unisim.vcomponents.LUT6
    generic map(
      INIT => X"AFA0CFCFAFA0C0C0"
    )
        port map (
      I0 => \alu_out[30]_INST_0_i_5_n_0\,
      I1 => \alu_out[31]_INST_0_i_8_n_0\,
      I2 => alu_b(1),
      I3 => \alu_out[31]_INST_0_i_14_n_0\,
      I4 => alu_b(2),
      I5 => \alu_out[31]_INST_0_i_15_n_0\,
      O => \alu_out[30]_INST_0_i_4_n_0\
    );
\alu_out[30]_INST_0_i_5\: unisim.vcomponents.LUT5
    generic map(
      INIT => X"30BB3088"
    )
        port map (
      I0 => alu_a(15),
      I1 => alu_b(3),
      I2 => alu_a(7),
      I3 => alu_b(4),
      I4 => alu_a(23),
      O => \alu_out[30]_INST_0_i_5_n_0\
    );
\alu_out[31]_INST_0_i_1\: unisim.vcomponents.LUT6
    generic map(
      INIT => X"AAA888A822200020"
    )
        port map (
      I0 => \alu_out[31]_INST_0_i_6_n_0\,
      I1 => alu_b(1),
      I2 => \alu_out[31]_INST_0_i_7_n_0\,
      I3 => alu_b(2),
      I4 => \alu_out[31]_INST_0_i_8_n_0\,
      I5 => \alu_out[31]_INST_0_i_9_n_0\,
      O => \alu_out[31]_INST_0_i_1_n_0\
    );
\alu_out[31]_INST_0_i_10\: unisim.vcomponents.LUT6
    generic map(
      INIT => X"AFA0CFCFAFA0C0C0"
    )
        port map (
      I0 => alu_a(0),
      I1 => alu_a(16),
      I2 => alu_b(3),
      I3 => alu_a(8),
      I4 => alu_b(4),
      I5 => alu_a(24),
      O => \alu_out[31]_INST_0_i_10_n_0\
    );
\alu_out[31]_INST_0_i_11\: unisim.vcomponents.LUT6
    generic map(
      INIT => X"CFC0AFAFCFC0A0A0"
    )
        port map (
      I0 => alu_a(20),
      I1 => alu_a(4),
      I2 => alu_b(3),
      I3 => alu_a(12),
      I4 => alu_b(4),
      I5 => alu_a(28),
      O => \alu_out[31]_INST_0_i_11_n_0\
    );
\alu_out[31]_INST_0_i_12\: unisim.vcomponents.LUT6
    generic map(
      INIT => X"AFA0CFCFAFA0C0C0"
    )
        port map (
      I0 => alu_a(2),
      I1 => alu_a(18),
      I2 => alu_b(3),
      I3 => alu_a(10),
      I4 => alu_b(4),
      I5 => alu_a(26),
      O => \alu_out[31]_INST_0_i_12_n_0\
    );
\alu_out[31]_INST_0_i_13\: unisim.vcomponents.LUT6
    generic map(
      INIT => X"AFA0CFCFAFA0C0C0"
    )
        port map (
      I0 => alu_a(6),
      I1 => alu_a(22),
      I2 => alu_b(3),
      I3 => alu_a(14),
      I4 => alu_b(4),
      I5 => alu_a(30),
      O => \alu_out[31]_INST_0_i_13_n_0\
    );
\alu_out[31]_INST_0_i_14\: unisim.vcomponents.LUT6
    generic map(
      INIT => X"AFA0CFCFAFA0C0C0"
    )
        port map (
      I0 => alu_a(1),
      I1 => alu_a(17),
      I2 => alu_b(3),
      I3 => alu_a(9),
      I4 => alu_b(4),
      I5 => alu_a(25),
      O => \alu_out[31]_INST_0_i_14_n_0\
    );
\alu_out[31]_INST_0_i_15\: unisim.vcomponents.LUT6
    generic map(
      INIT => X"AFA0CFCFAFA0C0C0"
    )
        port map (
      I0 => alu_a(5),
      I1 => alu_a(21),
      I2 => alu_b(3),
      I3 => alu_a(13),
      I4 => alu_b(4),
      I5 => alu_a(29),
      O => \alu_out[31]_INST_0_i_15_n_0\
    );
\alu_out[31]_INST_0_i_2\: unisim.vcomponents.LUT2
    generic map(
      INIT => X"8"
    )
        port map (
      I0 => alu_a(31),
      I1 => alu_sel(0),
      O => \alu_out[31]_INST_0_i_2_n_0\
    );
\alu_out[31]_INST_0_i_3\: unisim.vcomponents.LUT6
    generic map(
      INIT => X"AFA0CFCFAFA0C0C0"
    )
        port map (
      I0 => \alu_out[31]_INST_0_i_10_n_0\,
      I1 => \alu_out[31]_INST_0_i_11_n_0\,
      I2 => alu_b(1),
      I3 => \alu_out[31]_INST_0_i_12_n_0\,
      I4 => alu_b(2),
      I5 => \alu_out[31]_INST_0_i_13_n_0\,
      O => \alu_out[31]_INST_0_i_3_n_0\
    );
\alu_out[31]_INST_0_i_4\: unisim.vcomponents.LUT2
    generic map(
      INIT => X"2"
    )
        port map (
      I0 => alu_b(0),
      I1 => alu_sel(0),
      O => \alu_out[31]_INST_0_i_4_n_0\
    );
\alu_out[31]_INST_0_i_6\: unisim.vcomponents.LUT2
    generic map(
      INIT => X"1"
    )
        port map (
      I0 => alu_b(0),
      I1 => alu_sel(0),
      O => \alu_out[31]_INST_0_i_6_n_0\
    );
\alu_out[31]_INST_0_i_7\: unisim.vcomponents.LUT6
    generic map(
      INIT => X"AFA0CFCFAFA0C0C0"
    )
        port map (
      I0 => alu_a(7),
      I1 => alu_a(23),
      I2 => alu_b(3),
      I3 => alu_a(15),
      I4 => alu_b(4),
      I5 => alu_a(31),
      O => \alu_out[31]_INST_0_i_7_n_0\
    );
\alu_out[31]_INST_0_i_8\: unisim.vcomponents.LUT6
    generic map(
      INIT => X"AFA0CFCFAFA0C0C0"
    )
        port map (
      I0 => alu_a(3),
      I1 => alu_a(19),
      I2 => alu_b(3),
      I3 => alu_a(11),
      I4 => alu_b(4),
      I5 => alu_a(27),
      O => \alu_out[31]_INST_0_i_8_n_0\
    );
\alu_out[31]_INST_0_i_9\: unisim.vcomponents.LUT3
    generic map(
      INIT => X"B8"
    )
        port map (
      I0 => \alu_out[31]_INST_0_i_14_n_0\,
      I1 => alu_b(2),
      I2 => \alu_out[31]_INST_0_i_15_n_0\,
      O => \alu_out[31]_INST_0_i_9_n_0\
    );
\alu_out[3]_INST_0_i_1\: unisim.vcomponents.LUT5
    generic map(
      INIT => X"33373F37"
    )
        port map (
      I0 => \alu_out[4]_INST_0_i_4_n_0\,
      I1 => alu_sel(1),
      I2 => alu_sel(0),
      I3 => alu_b(0),
      I4 => \alu_out[3]_INST_0_i_4_n_0\,
      O => \alu_out[3]_INST_0_i_1_n_0\
    );
\alu_out[3]_INST_0_i_2\: unisim.vcomponents.LUT6
    generic map(
      INIT => X"FF33CC00B8B8B8B8"
    )
        port map (
      I0 => \alu_out[7]_INST_0_i_5_n_0\,
      I1 => alu_b(2),
      I2 => \alu_out[3]_INST_0_i_5_n_0\,
      I3 => \alu_out[9]_INST_0_i_5_n_0\,
      I4 => \alu_out[5]_INST_0_i_5_n_0\,
      I5 => alu_b(1),
      O => \alu_out[3]_INST_0_i_2_n_0\
    );
\alu_out[3]_INST_0_i_4\: unisim.vcomponents.LUT6
    generic map(
      INIT => X"FFFFFFFFFFF4FFF7"
    )
        port map (
      I0 => alu_a(0),
      I1 => alu_b(1),
      I2 => alu_b(2),
      I3 => alu_b(4),
      I4 => alu_a(2),
      I5 => alu_b(3),
      O => \alu_out[3]_INST_0_i_4_n_0\
    );
\alu_out[3]_INST_0_i_5\: unisim.vcomponents.LUT6
    generic map(
      INIT => X"AFA0CFCFAFA0C0C0"
    )
        port map (
      I0 => alu_a(27),
      I1 => alu_a(11),
      I2 => alu_b(3),
      I3 => alu_a(19),
      I4 => alu_b(4),
      I5 => alu_a(3),
      O => \alu_out[3]_INST_0_i_5_n_0\
    );
\alu_out[4]_INST_0_i_1\: unisim.vcomponents.LUT5
    generic map(
      INIT => X"33373F37"
    )
        port map (
      I0 => \alu_out[5]_INST_0_i_4_n_0\,
      I1 => alu_sel(1),
      I2 => alu_sel(0),
      I3 => alu_b(0),
      I4 => \alu_out[4]_INST_0_i_4_n_0\,
      O => \alu_out[4]_INST_0_i_1_n_0\
    );
\alu_out[4]_INST_0_i_2\: unisim.vcomponents.LUT6
    generic map(
      INIT => X"AFA0CFCFAFA0C0C0"
    )
        port map (
      I0 => \alu_out[10]_INST_0_i_5_n_0\,
      I1 => \alu_out[6]_INST_0_i_5_n_0\,
      I2 => alu_b(1),
      I3 => \alu_out[8]_INST_0_i_5_n_0\,
      I4 => alu_b(2),
      I5 => \alu_out[4]_INST_0_i_5_n_0\,
      O => \alu_out[4]_INST_0_i_2_n_0\
    );
\alu_out[4]_INST_0_i_4\: unisim.vcomponents.LUT6
    generic map(
      INIT => X"FFFFFFFFFFF4FFF7"
    )
        port map (
      I0 => alu_a(1),
      I1 => alu_b(1),
      I2 => alu_b(2),
      I3 => alu_b(3),
      I4 => alu_a(3),
      I5 => alu_b(4),
      O => \alu_out[4]_INST_0_i_4_n_0\
    );
\alu_out[4]_INST_0_i_5\: unisim.vcomponents.LUT6
    generic map(
      INIT => X"AFA0CFCFAFA0C0C0"
    )
        port map (
      I0 => alu_a(28),
      I1 => alu_a(12),
      I2 => alu_b(3),
      I3 => alu_a(20),
      I4 => alu_b(4),
      I5 => alu_a(4),
      O => \alu_out[4]_INST_0_i_5_n_0\
    );
\alu_out[5]_INST_0_i_1\: unisim.vcomponents.LUT5
    generic map(
      INIT => X"33373F37"
    )
        port map (
      I0 => \alu_out[6]_INST_0_i_4_n_0\,
      I1 => alu_sel(1),
      I2 => alu_sel(0),
      I3 => alu_b(0),
      I4 => \alu_out[5]_INST_0_i_4_n_0\,
      O => \alu_out[5]_INST_0_i_1_n_0\
    );
\alu_out[5]_INST_0_i_2\: unisim.vcomponents.LUT6
    generic map(
      INIT => X"AFA0CFCFAFA0C0C0"
    )
        port map (
      I0 => \alu_out[11]_INST_0_i_5_n_0\,
      I1 => \alu_out[7]_INST_0_i_5_n_0\,
      I2 => alu_b(1),
      I3 => \alu_out[9]_INST_0_i_5_n_0\,
      I4 => alu_b(2),
      I5 => \alu_out[5]_INST_0_i_5_n_0\,
      O => \alu_out[5]_INST_0_i_2_n_0\
    );
\alu_out[5]_INST_0_i_4\: unisim.vcomponents.LUT6
    generic map(
      INIT => X"FFEFFFFFFFEF0000"
    )
        port map (
      I0 => alu_b(2),
      I1 => alu_b(4),
      I2 => alu_a(2),
      I3 => alu_b(3),
      I4 => alu_b(1),
      I5 => \alu_out[7]_INST_0_i_6_n_0\,
      O => \alu_out[5]_INST_0_i_4_n_0\
    );
\alu_out[5]_INST_0_i_5\: unisim.vcomponents.LUT6
    generic map(
      INIT => X"AFA0CFCFAFA0C0C0"
    )
        port map (
      I0 => alu_a(29),
      I1 => alu_a(13),
      I2 => alu_b(3),
      I3 => alu_a(21),
      I4 => alu_b(4),
      I5 => alu_a(5),
      O => \alu_out[5]_INST_0_i_5_n_0\
    );
\alu_out[6]_INST_0_i_1\: unisim.vcomponents.LUT5
    generic map(
      INIT => X"33373F37"
    )
        port map (
      I0 => \alu_out[7]_INST_0_i_4_n_0\,
      I1 => alu_sel(1),
      I2 => alu_sel(0),
      I3 => alu_b(0),
      I4 => \alu_out[6]_INST_0_i_4_n_0\,
      O => \alu_out[6]_INST_0_i_1_n_0\
    );
\alu_out[6]_INST_0_i_2\: unisim.vcomponents.LUT6
    generic map(
      INIT => X"AFA0CFCFAFA0C0C0"
    )
        port map (
      I0 => \alu_out[12]_INST_0_i_5_n_0\,
      I1 => \alu_out[8]_INST_0_i_5_n_0\,
      I2 => alu_b(1),
      I3 => \alu_out[10]_INST_0_i_5_n_0\,
      I4 => alu_b(2),
      I5 => \alu_out[6]_INST_0_i_5_n_0\,
      O => \alu_out[6]_INST_0_i_2_n_0\
    );
\alu_out[6]_INST_0_i_4\: unisim.vcomponents.LUT6
    generic map(
      INIT => X"FFEFFFFFFFEF0000"
    )
        port map (
      I0 => alu_b(2),
      I1 => alu_b(3),
      I2 => alu_a(3),
      I3 => alu_b(4),
      I4 => alu_b(1),
      I5 => \alu_out[8]_INST_0_i_6_n_0\,
      O => \alu_out[6]_INST_0_i_4_n_0\
    );
\alu_out[6]_INST_0_i_5\: unisim.vcomponents.LUT6
    generic map(
      INIT => X"AFA0CFCFAFA0C0C0"
    )
        port map (
      I0 => alu_a(30),
      I1 => alu_a(14),
      I2 => alu_b(3),
      I3 => alu_a(22),
      I4 => alu_b(4),
      I5 => alu_a(6),
      O => \alu_out[6]_INST_0_i_5_n_0\
    );
\alu_out[7]_INST_0_i_1\: unisim.vcomponents.LUT5
    generic map(
      INIT => X"33373F37"
    )
        port map (
      I0 => \alu_out[8]_INST_0_i_4_n_0\,
      I1 => alu_sel(1),
      I2 => alu_sel(0),
      I3 => alu_b(0),
      I4 => \alu_out[7]_INST_0_i_4_n_0\,
      O => \alu_out[7]_INST_0_i_1_n_0\
    );
\alu_out[7]_INST_0_i_2\: unisim.vcomponents.LUT6
    generic map(
      INIT => X"AFA0CFCFAFA0C0C0"
    )
        port map (
      I0 => \alu_out[13]_INST_0_i_5_n_0\,
      I1 => \alu_out[9]_INST_0_i_5_n_0\,
      I2 => alu_b(1),
      I3 => \alu_out[11]_INST_0_i_5_n_0\,
      I4 => alu_b(2),
      I5 => \alu_out[7]_INST_0_i_5_n_0\,
      O => \alu_out[7]_INST_0_i_2_n_0\
    );
\alu_out[7]_INST_0_i_4\: unisim.vcomponents.LUT3
    generic map(
      INIT => X"B8"
    )
        port map (
      I0 => \alu_out[7]_INST_0_i_6_n_0\,
      I1 => alu_b(1),
      I2 => \alu_out[9]_INST_0_i_6_n_0\,
      O => \alu_out[7]_INST_0_i_4_n_0\
    );
\alu_out[7]_INST_0_i_5\: unisim.vcomponents.LUT6
    generic map(
      INIT => X"AFA0CFCFAFA0C0C0"
    )
        port map (
      I0 => alu_a(31),
      I1 => alu_a(15),
      I2 => alu_b(3),
      I3 => alu_a(23),
      I4 => alu_b(4),
      I5 => alu_a(7),
      O => \alu_out[7]_INST_0_i_5_n_0\
    );
\alu_out[7]_INST_0_i_6\: unisim.vcomponents.LUT5
    generic map(
      INIT => X"FFFFF4F7"
    )
        port map (
      I0 => alu_a(0),
      I1 => alu_b(2),
      I2 => alu_b(3),
      I3 => alu_a(4),
      I4 => alu_b(4),
      O => \alu_out[7]_INST_0_i_6_n_0\
    );
\alu_out[8]_INST_0_i_1\: unisim.vcomponents.LUT5
    generic map(
      INIT => X"33373F37"
    )
        port map (
      I0 => \alu_out[9]_INST_0_i_4_n_0\,
      I1 => alu_sel(1),
      I2 => alu_sel(0),
      I3 => alu_b(0),
      I4 => \alu_out[8]_INST_0_i_4_n_0\,
      O => \alu_out[8]_INST_0_i_1_n_0\
    );
\alu_out[8]_INST_0_i_2\: unisim.vcomponents.LUT6
    generic map(
      INIT => X"AFA0CFCFAFA0C0C0"
    )
        port map (
      I0 => \alu_out[14]_INST_0_i_5_n_0\,
      I1 => \alu_out[10]_INST_0_i_5_n_0\,
      I2 => alu_b(1),
      I3 => \alu_out[12]_INST_0_i_5_n_0\,
      I4 => alu_b(2),
      I5 => \alu_out[8]_INST_0_i_5_n_0\,
      O => \alu_out[8]_INST_0_i_2_n_0\
    );
\alu_out[8]_INST_0_i_4\: unisim.vcomponents.LUT3
    generic map(
      INIT => X"B8"
    )
        port map (
      I0 => \alu_out[8]_INST_0_i_6_n_0\,
      I1 => alu_b(1),
      I2 => \alu_out[10]_INST_0_i_6_n_0\,
      O => \alu_out[8]_INST_0_i_4_n_0\
    );
\alu_out[8]_INST_0_i_5\: unisim.vcomponents.LUT6
    generic map(
      INIT => X"AFA0CFCFAFA0C0C0"
    )
        port map (
      I0 => alu_a(31),
      I1 => alu_a(16),
      I2 => alu_b(3),
      I3 => alu_a(24),
      I4 => alu_b(4),
      I5 => alu_a(8),
      O => \alu_out[8]_INST_0_i_5_n_0\
    );
\alu_out[8]_INST_0_i_6\: unisim.vcomponents.LUT5
    generic map(
      INIT => X"FFFFF4F7"
    )
        port map (
      I0 => alu_a(1),
      I1 => alu_b(2),
      I2 => alu_b(3),
      I3 => alu_a(5),
      I4 => alu_b(4),
      O => \alu_out[8]_INST_0_i_6_n_0\
    );
\alu_out[9]_INST_0_i_1\: unisim.vcomponents.LUT5
    generic map(
      INIT => X"33373F37"
    )
        port map (
      I0 => \alu_out[10]_INST_0_i_4_n_0\,
      I1 => alu_sel(1),
      I2 => alu_sel(0),
      I3 => alu_b(0),
      I4 => \alu_out[9]_INST_0_i_4_n_0\,
      O => \alu_out[9]_INST_0_i_1_n_0\
    );
\alu_out[9]_INST_0_i_2\: unisim.vcomponents.LUT6
    generic map(
      INIT => X"AFA0CFCFAFA0C0C0"
    )
        port map (
      I0 => \alu_out[15]_INST_0_i_6_n_0\,
      I1 => \alu_out[11]_INST_0_i_5_n_0\,
      I2 => alu_b(1),
      I3 => \alu_out[13]_INST_0_i_5_n_0\,
      I4 => alu_b(2),
      I5 => \alu_out[9]_INST_0_i_5_n_0\,
      O => \alu_out[9]_INST_0_i_2_n_0\
    );
\alu_out[9]_INST_0_i_4\: unisim.vcomponents.LUT3
    generic map(
      INIT => X"B8"
    )
        port map (
      I0 => \alu_out[9]_INST_0_i_6_n_0\,
      I1 => alu_b(1),
      I2 => \alu_out[11]_INST_0_i_6_n_0\,
      O => \alu_out[9]_INST_0_i_4_n_0\
    );
\alu_out[9]_INST_0_i_5\: unisim.vcomponents.LUT6
    generic map(
      INIT => X"AFA0CFCFAFA0C0C0"
    )
        port map (
      I0 => alu_a(31),
      I1 => alu_a(17),
      I2 => alu_b(3),
      I3 => alu_a(25),
      I4 => alu_b(4),
      I5 => alu_a(9),
      O => \alu_out[9]_INST_0_i_5_n_0\
    );
\alu_out[9]_INST_0_i_6\: unisim.vcomponents.LUT5
    generic map(
      INIT => X"FFF4FFF7"
    )
        port map (
      I0 => alu_a(2),
      I1 => alu_b(2),
      I2 => alu_b(3),
      I3 => alu_b(4),
      I4 => alu_a(6),
      O => \alu_out[9]_INST_0_i_6_n_0\
    );
inst: entity work.decalper_eb_ot_sdeen_pot_pi_dehcac_xnilix_alu32
     port map (
      alu_a(31 downto 0) => alu_a(31 downto 0),
      alu_b(31 downto 0) => alu_b(31 downto 0),
      alu_out(31 downto 0) => alu_out(31 downto 0),
      \alu_out[11]_0\ => \alu_out[11]_INST_0_i_2_n_0\,
      \alu_out[11]_1\ => \alu_out[12]_INST_0_i_2_n_0\,
      \alu_out[13]_0\ => \alu_out[13]_INST_0_i_2_n_0\,
      \alu_out[13]_1\ => \alu_out[14]_INST_0_i_2_n_0\,
      \alu_out[15]_0\ => \alu_out[15]_INST_0_i_2_n_0\,
      \alu_out[15]_1\ => \alu_out[16]_INST_0_i_2_n_0\,
      \alu_out[17]_0\ => \alu_out[17]_INST_0_i_2_n_0\,
      \alu_out[17]_1\ => \alu_out[18]_INST_0_i_2_n_0\,
      \alu_out[18]_0\ => \alu_out[19]_INST_0_i_2_n_0\,
      \alu_out[19]_0\ => \alu_out[20]_INST_0_i_2_n_0\,
      \alu_out[1]_0\ => \alu_out[1]_INST_0_i_2_n_0\,
      \alu_out[1]_1\ => \alu_out[2]_INST_0_i_2_n_0\,
      \alu_out[20]_0\ => \alu_out[21]_INST_0_i_2_n_0\,
      \alu_out[21]_0\ => \alu_out[22]_INST_0_i_2_n_0\,
      \alu_out[22]_0\ => \alu_out[23]_INST_0_i_2_n_0\,
      \alu_out[23]_0\ => \alu_out[24]_INST_0_i_2_n_0\,
      \alu_out[24]_0\ => \alu_out[25]_INST_0_i_2_n_0\,
      \alu_out[25]_0\ => \alu_out[26]_INST_0_i_2_n_0\,
      \alu_out[26]_0\ => \alu_out[27]_INST_0_i_2_n_0\,
      \alu_out[27]_0\ => \alu_out[28]_INST_0_i_2_n_0\,
      \alu_out[28]_0\ => \alu_out[29]_INST_0_i_2_n_0\,
      \alu_out[29]_0\ => \alu_out[30]_INST_0_i_2_n_0\,
      \alu_out[2]_0\ => \alu_out[3]_INST_0_i_2_n_0\,
      \alu_out[31]_0\ => \alu_out[31]_INST_0_i_2_n_0\,
      \alu_out[31]_1\ => \alu_out[31]_INST_0_i_3_n_0\,
      \alu_out[31]_2\ => \alu_out[31]_INST_0_i_4_n_0\,
      \alu_out[4]_0\ => \alu_out[4]_INST_0_i_2_n_0\,
      \alu_out[4]_1\ => \alu_out[5]_INST_0_i_2_n_0\,
      \alu_out[5]_0\ => \alu_out[6]_INST_0_i_2_n_0\,
      \alu_out[6]_0\ => \alu_out[7]_INST_0_i_2_n_0\,
      \alu_out[7]_0\ => \alu_out[8]_INST_0_i_2_n_0\,
      \alu_out[8]_0\ => \alu_out[9]_INST_0_i_2_n_0\,
      \alu_out[9]_0\ => \alu_out[10]_INST_0_i_2_n_0\,
      alu_out_0_sp_1 => \alu_out[0]_INST_0_i_1_n_0\,
      alu_out_10_sp_1 => \alu_out[10]_INST_0_i_1_n_0\,
      alu_out_11_sp_1 => \alu_out[11]_INST_0_i_1_n_0\,
      alu_out_12_sp_1 => \alu_out[12]_INST_0_i_1_n_0\,
      alu_out_13_sp_1 => \alu_out[13]_INST_0_i_1_n_0\,
      alu_out_14_sp_1 => \alu_out[14]_INST_0_i_1_n_0\,
      alu_out_15_sp_1 => \alu_out[15]_INST_0_i_1_n_0\,
      alu_out_16_sp_1 => \alu_out[16]_INST_0_i_1_n_0\,
      alu_out_17_sp_1 => \alu_out[17]_INST_0_i_1_n_0\,
      alu_out_18_sp_1 => \alu_out[18]_INST_0_i_1_n_0\,
      alu_out_19_sp_1 => \alu_out[19]_INST_0_i_1_n_0\,
      alu_out_1_sp_1 => \alu_out[1]_INST_0_i_1_n_0\,
      alu_out_20_sp_1 => \alu_out[20]_INST_0_i_1_n_0\,
      alu_out_21_sp_1 => \alu_out[21]_INST_0_i_1_n_0\,
      alu_out_22_sp_1 => \alu_out[22]_INST_0_i_1_n_0\,
      alu_out_23_sp_1 => \alu_out[23]_INST_0_i_1_n_0\,
      alu_out_24_sp_1 => \alu_out[24]_INST_0_i_1_n_0\,
      alu_out_25_sp_1 => \alu_out[25]_INST_0_i_1_n_0\,
      alu_out_26_sp_1 => \alu_out[26]_INST_0_i_1_n_0\,
      alu_out_27_sp_1 => \alu_out[27]_INST_0_i_1_n_0\,
      alu_out_28_sp_1 => \alu_out[28]_INST_0_i_1_n_0\,
      alu_out_29_sp_1 => \alu_out[29]_INST_0_i_1_n_0\,
      alu_out_2_sp_1 => \alu_out[2]_INST_0_i_1_n_0\,
      alu_out_30_sp_1 => \alu_out[30]_INST_0_i_1_n_0\,
      alu_out_31_sp_1 => \alu_out[31]_INST_0_i_1_n_0\,
      alu_out_3_sp_1 => \alu_out[3]_INST_0_i_1_n_0\,
      alu_out_4_sp_1 => \alu_out[4]_INST_0_i_1_n_0\,
      alu_out_5_sp_1 => \alu_out[5]_INST_0_i_1_n_0\,
      alu_out_6_sp_1 => \alu_out[6]_INST_0_i_1_n_0\,
      alu_out_7_sp_1 => \alu_out[7]_INST_0_i_1_n_0\,
      alu_out_8_sp_1 => \alu_out[8]_INST_0_i_1_n_0\,
      alu_out_9_sp_1 => \alu_out[9]_INST_0_i_1_n_0\,
      alu_sel(2 downto 0) => alu_sel(2 downto 0)
    );
end STRUCTURE;
