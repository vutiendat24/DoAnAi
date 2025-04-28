import { useState, useEffect } from 'react';

const BOARD_SIZE = 15;
const TIME_LIMIT = 20; // 20 giây mỗi lượt

function App() {
  const emptyBoard = Array.from({ length: BOARD_SIZE }, () => Array(BOARD_SIZE).fill(0));

  const [board, setBoard] = useState(emptyBoard);
  const [currentPlayer, setCurrentPlayer] = useState(1); // 1: Người (X), -1: Máy (O)
  const [winningCells, setWinningCells] = useState([]);
  const [timer, setTimer] = useState(TIME_LIMIT);

  useEffect(() => {
    if (winningCells.length > 0) return;

    // Máy tự đi khi tới lượt máy (O)
    if (currentPlayer === -1) {
      const delay = setTimeout(() => {
        makeAIMove();
      }, 500); // delay 0.5s cho tự nhiên

      return () => clearTimeout(delay);
    }
  }, [currentPlayer, winningCells]);

  useEffect(() => {
    if (winningCells.length > 0) return; // Nếu thắng thì dừng đếm thời gian

    const interval = setInterval(() => {
      setTimer((prev) => {
        if (prev === 1) {
          handleTimeout();
          return TIME_LIMIT;
        }
        return prev - 1;
      });
    }, 1000);

    return () => clearInterval(interval);
  }, [currentPlayer, winningCells]);

  const handleTimeout = () => {
    alert(`Hết giờ! Người chơi ${currentPlayer === 1 ? 'X' : 'O'} bị mất lượt.`);
    setCurrentPlayer((prev) => prev * -1); // Chuyển lượt
    setTimer(TIME_LIMIT);
  };

  const handleClick = (row, col) => {
    if (board[row][col] !== 0 || winningCells.length > 0 || currentPlayer !== 1) return; // Chỉ người chơi mới được click

    makeMove(row, col, 1);
  };

  const makeMove = (row, col, player) => {
    const newBoard = board.map((r) => [...r]);
    newBoard[row][col] = player;
    setBoard(newBoard);

    const winPositions = isWin(newBoard, row, col, player);
    if (winPositions) {
      setWinningCells(winPositions);
      alert(`Người chơi ${player === 1 ? 'X' : 'O'} thắng!`);
      return;
    }

    setCurrentPlayer(player * -1);
    setTimer(TIME_LIMIT);
  };

  const makeAIMove = async () => {
    try {
      const response = await fetch('http://localhost:8000/predict', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ board }),
      });
      const data = await response.json();
      const { row, col } = data;

      if (row !== -1 && col !== -1) {
        makeMove(row, col, -1);
      }
    } catch (error) {
      console.error('Lỗi gọi AI server:', error);
    }
  };

  const handleRestart = () => {
    setBoard(emptyBoard);
    setCurrentPlayer(1);
    setWinningCells([]);
    setTimer(TIME_LIMIT);
  };

  return (
    <div className="min-h-screen bg-gradient-to-br from-blue-100 to-gray-100 flex flex-col items-center justify-center p-4">
      <h1 className="text-4xl font-bold text-blue-800 mb-6">Caro 15x15 (Người vs Máy)</h1>

      <div className="flex flex-col md:flex-row items-center gap-8">
        {/* Thông tin game */}
        <div className="bg-white rounded-lg shadow-lg p-6 w-full md:w-64 text-center">
          <h2 className="text-xl font-semibold text-gray-800 mb-2">
            Lượt: {currentPlayer === 1 ? 'Người (X)' : 'Máy (O)'}
          </h2>
          <h2 className="text-xl font-semibold text-gray-800">
            Thời gian: <span className={timer <= 5 ? 'text-red-600' : 'text-green-600'}>{timer}s</span>
          </h2>
          {winningCells.length > 0 && (
            <p className="mt-4 text-lg font-bold text-blue-600">
              {currentPlayer === 1 ? 'Máy (O)' : 'Người (X)'} thắng!
            </p>
          )}
          <button
            onClick={handleRestart}
            className="mt-4 px-4 py-2 bg-blue-600 text-white rounded-lg hover:bg-blue-700 transition duration-200"
          >
            Chơi lại
          </button>
        </div>

        {/* Bàn cờ */}
        <div className="bg-white rounded-lg shadow-lg p-4">
          <div className="grid grid-cols-15 gap-0">
            {board.map((row, rowIndex) =>
              row.map((cell, colIndex) => (
                <div
                  key={`${rowIndex}-${colIndex}`}
                  className={`
                    w-10 h-10 flex items-center justify-center border border-gray-300 text-2xl font-bold
                    ${winningCells.some(([r, c]) => r === rowIndex && c === colIndex) ? 'bg-yellow-300' : 'bg-gray-50'}
                    ${cell === 0 && currentPlayer === 1 ? 'hover:bg-blue-100 cursor-pointer' : 'cursor-default'}
                    ${cell === 1 ? 'text-blue-600' : cell === -1 ? 'text-red-600' : ''}
                  `}
                  onClick={() => handleClick(rowIndex, colIndex)}
                >
                  {cell === 1 ? 'X' : cell === -1 ? 'O' : ''}
                </div>
              ))
            )}
          </div>
        </div>
      </div>
    </div>
  );
}

function isWin(board, row, col, player) {
  const directions = [
    { dr: 0, dc: 1 },
    { dr: 1, dc: 0 },
    { dr: 1, dc: 1 },
    { dr: 1, dc: -1 },
  ];

  const BOARD_SIZE = board.length;

  for (let { dr, dc } of directions) {
    let positions = [[row, col]];

    let r = row + dr;
    let c = col + dc;
    while (r >= 0 && r < BOARD_SIZE && c >= 0 && c < BOARD_SIZE && board[r][c] === player) {
      positions.push([r, c]);
      r += dr;
      c += dc;
    }

    r = row - dr;
    c = col - dc;
    while (r >= 0 && r < BOARD_SIZE && c >= 0 && c < BOARD_SIZE && board[r][c] === player) {
      positions.push([r, c]);
      r -= dr;
      c -= dc;
    }

    if (positions.length >= 5) {
      return positions;
    }
  }

  return null;
}

export default App;