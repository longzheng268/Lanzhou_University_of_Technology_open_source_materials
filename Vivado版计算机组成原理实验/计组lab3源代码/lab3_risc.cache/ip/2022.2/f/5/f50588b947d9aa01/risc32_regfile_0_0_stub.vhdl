-- Copyright 1986-2022 Xilinx, Inc. All Rights Reserved.
-- --------------------------------------------------------------------------------
-- Tool Version: Vivado v.2022.2 (win64) Build 3671981 Fri Oct 14 05:00:03 MDT 2022
-- Date        : Mon Oct 28 16:04:11 2024
-- Host        : cop running 64-bit major release  (build 9200)
-- Command     : write_vhdl -force -mode synth_stub -rename_top decalper_eb_ot_sdeen_pot_pi_dehcac_xnilix -prefix
--               decalper_eb_ot_sdeen_pot_pi_dehcac_xnilix_ risc32_regfile_0_0_stub.vhdl
-- Design      : risc32_regfile_0_0
-- Purpose     : Stub declaration of top-level module interface
-- Device      : xc7a100tcsg324-1
-- --------------------------------------------------------------------------------
library IEEE;
use IEEE.STD_LOGIC_1164.ALL;

entity decalper_eb_ot_sdeen_pot_pi_dehcac_xnilix is
  Port ( 
    clk : in STD_LOGIC;
    w_ena : in STD_LOGIC;
    r_addr1 : in STD_LOGIC_VECTOR ( 4 downto 0 );
    r_addr2 : in STD_LOGIC_VECTOR ( 4 downto 0 );
    w_addr : in STD_LOGIC_VECTOR ( 4 downto 0 );
    w_data : in STD_LOGIC_VECTOR ( 31 downto 0 );
    reg_data1 : out STD_LOGIC_VECTOR ( 31 downto 0 );
    reg_data2 : out STD_LOGIC_VECTOR ( 31 downto 0 )
  );

end decalper_eb_ot_sdeen_pot_pi_dehcac_xnilix;

architecture stub of decalper_eb_ot_sdeen_pot_pi_dehcac_xnilix is
attribute syn_black_box : boolean;
attribute black_box_pad_pin : string;
attribute syn_black_box of stub : architecture is true;
attribute black_box_pad_pin of stub : architecture is "clk,w_ena,r_addr1[4:0],r_addr2[4:0],w_addr[4:0],w_data[31:0],reg_data1[31:0],reg_data2[31:0]";
attribute X_CORE_INFO : string;
attribute X_CORE_INFO of stub : architecture is "regfile,Vivado 2022.2";
begin
end;
