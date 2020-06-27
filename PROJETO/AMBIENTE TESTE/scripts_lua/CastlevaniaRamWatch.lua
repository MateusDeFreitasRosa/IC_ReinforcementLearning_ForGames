SCRIPT_TITLE = "CASTLEVANIA_RAM_WATCH"
SCRIPT_VERSION = "0.01"

strMode = "turbo";

while true do

  -- Execute instructions for FCEUX
  --gui.text(50,50,gui.gdscreenshot());
  --emu.speedmode(strMode); -- speed up emulation (useful for training)
  --memory.readbyte(0x0003F);
  --memory.readbyte(0x00040);
  --joypad.read(playern); -- get the input table for the player who's input you want to read (a number!)
  --joypad.write(playern, inputtable); -- set the input for player n. Note that this will overwrite any input from the user, and only when this is used.
  --emu.wait() -- waits 1/60 sec (for python app, for example?)
  --joypad.set(int player, table input) -- Table keys look like this (case sensitive): up, down, left, right, A, B, start, select
  
  -- FCEUX Lua Docs: http://www.fceux.com/web/help/fceux.html?Commands.html
  -- FCEUX Lua Docs: http://www.fceux.com/web/help/fceux.html?LuaFunctionsList.html
  -- RAM Map: https://datacrystal.romhacking.net/wiki/Castlevania:RAM_map
  -- Goal: increase position X?
  gui.text(50,40, "pos Y = " .. memory.readbyte(0x0003F)); -- get position Y from Simon Belmont
  gui.text(50,50, "pos X = " .. memory.readbyte(0x00040)); -- get position X from Simon Belmont
  gui.text(50,60, "pt2 X = " .. memory.readbyte(0x00041)); -- accumulator for X from Simon Belmont (+1 on 0040 overflow)
  
  file_ready = io.open('lua_ready.txt', 'w');
  io.output(file_ready);  -- sets the default output file as cvram.txt
  io.write("can not");
  io.close(file_ready);

  file01 = io.open('lua_ready.txt', 'w');
  io.input(file01);
  lua_ready = io.read();
  io.close(file01);
  gui.text(50,70, "lua_ready = " .. lua_ready);

  

  if (lua_ready ~= 'can') then 
    file = io.open("cvram.txt", "a");  -- Opens a file in append mode
    io.output(file);  -- sets the default output file as cvram.txt
    io.write(memory.readbyte(0x0003F) .. " " .. memory.readbyte(0x00040) .. " " .. memory.readbyte(0x00041) .. "\n" ); -- appends a word test to the last line of the file
    io.close(file);
  
    file_ready = io.open('lua_ready.txt', 'w');
    io.output(file_ready);  -- sets the default output file as cvram.txt
    io.write("can");
    io.close(file_ready);
 

    file01 = io.open('lua_ready.txt', 'w');
    io.input(file01);
    lua_ready = io.read();
    io.close(file01);
    gui.text(50,70, "lua_ready = " .. lua_ready);
    
  
    gui.savescreenshotas("cv_screen.png");
  end
  

  emu.frameadvance() -- This essentially tells FCEUX to keep running

end