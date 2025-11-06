-- Copyright 1986-2022 Xilinx, Inc. All Rights Reserved.
-- --------------------------------------------------------------------------------
-- Tool Version: Vivado v.2022.2 (win64) Build 3671981 Fri Oct 14 05:00:03 MDT 2022
-- Date        : Mon Oct 28 16:04:11 2024
-- Host        : cop running 64-bit major release  (build 9200)
-- Command     : write_vhdl -force -mode synth_stub -rename_top decalper_eb_ot_sdeen_pot_pi_dehcac_xnilix -prefix
--               decalper_eb_ot_sdeen_pot_pi_dehcac_xnilix_ risc32_control_rom_0_0_stub.vhdl
-- Design      : risc32_control_rom_0_0
-- Purpose     : Stub declaration of top-level module interface
-- Device      : xc7a100tcsg324-1
-- --------------------------------------------------------------------------------
library IEEE;
use IEEE.STD_LOGIC_1164.ALL;

entity decalper_eb_ot_sdeen_pot_pi_dehcac_xnilix is
  Port ( 
    inst : in STD_LOGIC_VECTOR ( 8 downto 0 );
    imm_sel : out STD_LOGIC;
    bsel : out STD_LOGIC;
    mem_sel : out STD_LOGIC;
    alusel : out STD_LOGIC_VECTOR ( 2 downto 0 );
    w_ena : out STD_LOGIC;
    mem_ena : out STD_LOGIC
  );

end decalper_eb_ot_sdeen_pot_pi_dehcac_xnilix;

architecture stub of decalper_eb_ot_sdeen_pot_pi_dehcac_xnilix is
attribute syn_black_box : boolean;
attribute black_box_pad_pin : string;
attribute syn_black_box of stub : architecture is true;
attribute black_box_pad_pin of stub : architecture is "inst[8:0],imm_sel,bsel,mem_sel,alusel[2:0],w_ena,mem_ena";
attribute X_CORE_INFO : string;
attribute X_CORE_INFO of stub : architecture is "control_rom,Vivado 2022.2";
begin
end;
