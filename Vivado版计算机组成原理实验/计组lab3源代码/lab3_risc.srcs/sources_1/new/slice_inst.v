`timescale 1ns / 1ps
//////////////////////////////////////////////////////////////////////////////////
// Company: lzut
// Engineer: Yao bin
// 
// Create Date: 2024/10/21 16:10:05
// Design Name: 
// Module Name: slice_inst
// Project Name: 
// Target Devices: 
// Tool Versions: 
// Description: 
// 
// Dependencies: 
// 
// Revision:
// Revision 0.01 - File Created
// Additional Comments:
// 
//////////////////////////////////////////////////////////////////////////////////


module slice_inst(input     [31:0]  instruction,
                  output            inst_31     ,
                  output            inst_30     ,
                  output    [6:0]   inst_31_25  ,
                  output    [4:0]   inst_24_20  ,
                  output    [4:0]   inst_19_15  ,
                  output    [2:0]   inst_14_12  ,
                  output    [4:0]   inst_11_7   ,                  
                  output    [4:0]   inst_6_2    );

assign  inst_31     =   instruction[31]     ;
assign  inst_30     =   instruction[30]     ;
assign  inst_31_25  =   instruction[31:25]  ;
assign  inst_24_20  =   instruction[24:20]  ;
assign  inst_19_15  =   instruction[19:15]  ;
assign  inst_14_12  =   instruction[14:12]  ;
assign  inst_11_7   =   instruction[11:7]   ;
assign  inst_6_2    =   instruction[6:2]    ;

endmodule
