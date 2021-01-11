local socket = require("socket.core")
local json = require("json2020")

PLAYER_NUMBER = 1;

function connect(address, port, laddress, lport)
    local sock, err = socket.tcp()
    if not sock then return nil, err end
    if laddress then
        local res, err = sock:bind(laddress, lport, -1)
        if not res then return nil, err end
    end
    local res, err = sock:connect(address, port)
    if not res then return nil, err end
    return sock
end

function bind(host, port, backlog)
    local sock, err = socket.tcp()
    if not sock then return nil, err end
    sock:setoption("reuseaddr", true)
    local res, err = sock:bind(host, port)
    if not res then return nil, err end
    res, err = sock:listen(backlog)
    if not res then return nil, err end
    return sock
end

--sock, err = bind("127.0.0.1", 80, -1)
--print(sock, err)

function getMatrizScreen(params)
    downsample = params['down_sample'] or 1
    local xLen = params['len_max_x'] or 255
    local yLen = params['len_max_y'] or 239
    local xMin = params['len_min_x'] or 1
    local yMin = params['len_min_y'] or 1
    local grayscale = params['grayscale'] or false
    
    if (xLen > 255) then
        xLen = 255
    elseif (yLen > 239) then
        yLen = 239
    elseif (xMin < 1) then
        xMin = 1
    elseif (yMin < 1) then
        yMin = 1
    end

    if downsample < 0 then
        downsample = 1
    end

    local matriz = {}
    if grayscale then
        for j=yMin, yLen, downsample do
            local auxMatriz = {}
            for i=xMin, xLen, downsample do
                local r,g,b,palette = emu.getscreenpixel(i-1,j-1,true)
                table.insert( auxMatriz, 0.299*r + 0.587*g + 0.114*b)
            end
            table.insert( matriz, auxMatriz)
        end
    else
        for j=yMin, yLen, downsample do
            local auxMatriz = {}
            for i=xMin, xLen, downsample do
                local r,g,b,palette = emu.getscreenpixel(i-1,j-1,true)
                local pixelColor = {r,g,b}
                table.insert( auxMatriz, pixelColor)
            end
            table.insert( matriz, auxMatriz)
        end
    end
    

    return matriz
end

function sendMessage(message)
    --print(matrix)
    --print('SendMessage')
    local  operation = json.encode(message);
    --local compressed = assert(lualzw.compress(operation))
    --print('Compressed: '.. string.len( compressed ));
    --print('Len: '..tostring(string.len( operation )));
    sock2:send('json'..operation)
    
end

sock2, err2 = connect("127.0.0.1", 12345)
sock2:settimeout(0)
print("Connected", sock2, err2)


function read_memory()
    local playerPosition = ((memory.readbyte(0x006D)*255) + memory.readbyte(0x0086))
    local enemyPosition = (memory.readbyte(0x006E)*255 + memory.readbyte(0x0087))
    return {
        reward = playerPosition,
        endgame = memory.readbyte(0x000E),
    }
    
end

Operation = {
    menu = {
        getScreenShot = function (params)
            local matriz = getMatrizScreen(params)
            sendMessage({matriz=matriz})
        end,
        nextStep = function (params)
            joypadControll = joypad.get(PLAYER_NUMBER)
            joypadControll[params['press']] = true
            joypad.set(PLAYER_NUMBER, joypadControll)

            local mem = read_memory()

            local map = {
                newState = getMatrizScreen(params['screenshot_params']),
                endgame = mem.endgame,
                reward = mem.reward
            }
            sendMessage(json.encode(map))
        end
    }
}
function Operation:execute(operation)
    local decode_string = json.decode(operation)
    self.menu[decode_string['operation']](decode_string['params'])
end

function reciveCommands()
    local message, err, part = sock2:receive('*all')
    --print('Message: '..(message or 'nil'))
    --print('Error: '..(err or 'nil'))
    print('Part: '..(part or 'nil'))
    if not message then
        message = part
    end
    if message and string.len(message)>0 then
        --print('Message: '..message)
        --print('Error: '..err)
        --print('Part: '..part)
        Operation:execute(message)
        --print(message)
        --local recCommand = json.decode(message)
        --table.insert(commandsQueue, recCommand)
        --coroutine.resume(parseCommandCoroutine)
    end
end

function teste()
    print('Teste')
end

function main()
    while true do
        emu.frameadvance()
        reciveCommands()
        socket.sleep(0.25)
    end
end
--gui.register(reciveCommands)

main();