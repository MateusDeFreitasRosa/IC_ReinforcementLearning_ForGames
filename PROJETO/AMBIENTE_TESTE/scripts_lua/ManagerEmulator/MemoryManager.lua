local memoryAcceptedCharacters = {'+', '-', '/', '*', ')', '(', ' ', '0', '1', '2', '3', '4', '5', '6', '7', '8', '9', 'x', 'A', 'B', 'C', 'D', 'E', 'F'}
-- For security only accept string with limited characters, because we use loadstring that is similar to the EVAL in other languages.

local tableMap = {};

function readMemoryMap()
    local retorno = '"retorno":{';
    local first=true;

    for key, value in pairs(tableMap) do
        func = assert(loadstring("return " .. value)) -- In current versions use 'load' insted of 'loadstring'
        if first then
            retorno = retorno .. string.format( '"%s": %d', key, func());
            first=false;
        else
            retorno = retorno .. string.format( ',"%s": %d', key, func());
        end
    end
    return retorno..'}';
end

--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
-- REGISTER RETURN MAP.

function hasCharacterAccept(character)
    for _, value in pairs(memoryAcceptedCharacters) do
        if character == value then
            return true;
        end
    end
    return false;
end

function verifyStringWithAcceptedCharacters(value)
    for character in value:gmatch(".") do
        if (not hasCharacterAccept(character)) then
            error('Character is not accepted: '..character)
            return false;
        end
    end
    return true;
end

function safeMemory(tableM)
    for key, value in pairs(tableM) do
        if (not verifyStringWithAcceptedCharacters(value)) then
            return false;
        end
    end
    return true;
end

function formatValue(value)
    local newString="";
    local i = 1
    while i <= #value do
        if (string.sub( value, i, i+1) == "0x") then
            newString = newString.. string.format( "memory.readbyte(%s)", string.sub( value, i, i+5 ))
            i = i+6;
        else
            newString = newString.. string.sub( value, i, i );
            i = i+1
        end
    end
    return string.format("%s", newString);
end

function addTableMap(tableM)
    if(safeMemory(tableM)) then
        for key, value in pairs(tableM) do
            local formatedValue = formatValue(value);
            tableMap[key] = formatedValue;
            --print(formatedValue);
        end
    else
        print('Table: '.. tableM)
        print('Unsafe memory');
        error("Block operation. Only accept: +, -, (, ), /, *, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, x")
    end
end

