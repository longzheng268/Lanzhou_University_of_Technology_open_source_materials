//
module imem(input  	   [31:0] inst_addr,
			output reg [31:0] inst);
				
	wire [5:0] addr;
	
	assign addr[5:0] = inst_addr[5:0];
	
	always @(*) begin
		case (addr)
			0  : inst = 32'h70100093;   //write 01 to regfile, address equalls 01(addi);
			4  : inst = 32'h00200113;
			8  : inst = 32'h00300193;
			12 : inst = 32'h00400213;
			16 : inst = 32'h00500293;
			20 : inst = 32'h00600313;
			24 : inst = 32'h00700393;
			28 : inst = 32'h00800413;
			32 : inst = 32'h01008793;    //add 10 to rs01 , write sum to rd15(addi);
			36 : inst = 32'h00f08833;	//rs01 + rs15 -> rd16  (add);
			default : inst = 0;
		endcase
	end
	
endmodule
	
	
			

