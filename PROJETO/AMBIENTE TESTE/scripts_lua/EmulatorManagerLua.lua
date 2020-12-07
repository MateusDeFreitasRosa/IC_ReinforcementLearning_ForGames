local socket = require("socket.core")
local json = require("json")


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

function getMatrizScreen()
    local xLen = 255
    local yLen = 239
    local matriz = {}
    for i=0, xLen do
        matriz[i] = {}
        for j=0, yLen do
            matriz[i][j] = emu.getscreenpixel(i,j,true)
        end
    end
    return matriz
end

function sendMessage(message)
    local  operation = json.encode(message)
    print('SEND: '..operation)
    --sock2:send(operation)
end

sock2, err2 = connect("127.0.0.1", 12345)
sock2:settimeout(0)
print("Connected", sock2, err2)

Operation = {
    menu = {
        getScreenShot = function ()
            local matriz = getMatrizScreen()
            sendMessage({matriz = matriz})
        end
    }
}

function Operation:execute(operation)
    self.menu[operation]()
end

function reciveCommands()
    local message, err, part = sock2:receive("*all")
    if not message then
        message = part
    end
    if message and string.len(message)>0 then
        print('Message: '..message)
        print('Error: '..err)
        print('Part: '..part)
        Operation:execute(message)
        
        
        --print(message)
        --local recCommand = json.decode(message)
        --table.insert(commandsQueue, recCommand)
        --coroutine.resume(parseCommandCoroutine)
    end
end


function main()
    while true do
        reciveCommands()
        emu.frameadvance()
    end
end

main()