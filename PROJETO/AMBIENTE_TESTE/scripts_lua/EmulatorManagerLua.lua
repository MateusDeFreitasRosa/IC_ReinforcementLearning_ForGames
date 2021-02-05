local socket = require("socket.core")
local json = require("json")
PLAYER_NUMBER = 1;
local obj

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

    --local matriz = {}
    local matriz = "["
    if grayscale then
        for j=yMin, yLen, downsample do
            --local auxMatriz = {}
            local auxMatriz=''
            if j==yMin then
                auxMatriz = "["
            elseif j>yMin then
                auxMatriz = ",["
            end
            for i=xMin, xLen, downsample do
                local r,g,b,palette = emu.getscreenpixel(i-1,j-1,true)
                --table.insert( auxMatriz, 0.299*r + 0.587*g + 0.114*b)
                if i > xMin then
                    auxMatriz = auxMatriz..","..(0.299*r + 0.587*g + 0.114*b)
                elseif i == xMin then
                    auxMatriz = auxMatriz..(0.299*r + 0.587*g + 0.114*b)
                end
            end
            --table.insert( matriz, auxMatriz)
            matriz = matriz..auxMatriz.."]"
        end
    else
        for j=yMin, yLen, downsample do
            --local auxMatriz = {}
            local auxMatriz=''
            if j==yMin then
                auxMatriz = "["
            elseif j>yMin then
                auxMatriz = ",["
            end
            for i=xMin, xLen, downsample do
                local r,g,b,palette = emu.getscreenpixel(i-1,j-1,true)
                --local pixelColor = {r,g,b}
                local pixelColor = "["..r..","..g..","..b.."]"
                --table.insert( auxMatriz, pixelColor)
                if i > xMin then
                    auxMatriz = auxMatriz..","..(0.299*r + 0.587*g + 0.114*b)
                elseif i == xMin then
                    auxMatriz = auxMatriz..(0.299*r + 0.587*g + 0.114*b)
                end
            end
            --table.insert( matriz, auxMatriz)
            matriz = matriz..auxMatriz.."]"
        end
    end
    
    return string.format( '"matriz":%s]', matriz)
end

function sendMessage(message)
    --print(matrix)
    --print('SendMessage')
    --local operation = json.encode(message);
    --local compressed = assert(lualzw.compress(operation))
    --print('Compressed: '.. string.len( compressed ));
    --print('Len: '..tostring(string.len( operation )));
    sock2:send(message)
    
end

sock2, err2 = connect("127.0.0.1", 12345)
sock2:settimeout(0.001)
print("Connected", sock2, err2)


function read_memory()
    local playerPosition = ((memory.readbyte(0x006D)*255) + memory.readbyte(0x0086))
    return {
        reward = playerPosition,
        endgame = memory.readbyte(0x000E),
    }
end

Operation = {
    lastAction='',
    menu = {
        getScreenShot = function (params)
            local matriz = getMatrizScreen(params)
            sendMessage(matriz)
        end,
        nextStep = function (params)
            if params['action'] then
                joypadControll = joypad.get(PLAYER_NUMBER)
                if Operation.lastAction ~= '' then
                    joypadControll[Operation.lastAction] = false
                end
                joypadControll[params['action']] = true
                joypad.set(PLAYER_NUMBER, joypadControll)
                Operation.lastAction=params['action']

            end

            local mem = read_memory()
            --print(getMatrizScreen(params['screenshot_params']))

            local map = string.format( '{%s,"endgame":%d,"reward":%d}',getMatrizScreen(params['screenshot_params']), mem.endgame, mem.reward) 
            sendMessage(map)
            emu.frameadvance()
        end,
        reset = function (params)
            Operation:reset(params)
        end
    }
}
function Operation:execute(operation)
    local decode_string = json.decode(operation)
    self.menu[decode_string['operation']](decode_string['params'])
end

function Operation:reset(params)
    --local obj = savestate.create(1)
    savestate.load(obj)
    self.menu['nextStep'](params)
end

function reciveCommands()
    local message, err, part = sock2:receive('*all')
    --print('Message: '..(message or 'nil'))
    --print('Error: '..(err or 'nil'))
    --print('Part: '..(part or 'nil'))
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
    obj = savestate.create(1)
    savestate.save(obj)
    while true do
        reciveCommands()
    end
end
--gui.register(reciveCommands)

main();