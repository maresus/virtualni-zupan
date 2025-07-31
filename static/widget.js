(function() {
    // URL do vaše aplikacije na Renderju
    const chatUrl = "https://virtualni-zupan.onrender.com/";

    // 1. Ustvarimo CSS stile za naš widget
    const style = document.createElement('style');
    style.innerHTML = `
        #zupan-chat-bubble {
            position: fixed;
            bottom: 20px;
            right: 20px;
            width: 60px;
            height: 60px;
            background-color: #007bff;
            border-radius: 50%;
            display: flex;
            align-items: center;
            justify-content: center;
            cursor: pointer;
            box-shadow: 0 4px 8px rgba(0,0,0,0.2);
            z-index: 9998;
            transition: transform 0.2s;
        }
        #zupan-chat-bubble:hover {
            transform: scale(1.1);
        }
        #zupan-chat-bubble svg {
            width: 32px;
            height: 32px;
            fill: white;
        }
        #zupan-chat-widget {
            position: fixed;
            bottom: 90px;
            right: 20px;
            width: 350px;
            height: 500px;
            border: 1px solid #ddd;
            border-radius: 10px;
            box-shadow: 0 4px 12px rgba(0,0,0,0.15);
            overflow: hidden;
            display: none; /* Na začetku je skrit */
            flex-direction: column;
            background-color: white;
            z-index: 9999;
        }
        #zupan-chat-widget iframe {
            width: 100%;
            height: 100%;
            border: none;
        }
    `;
    document.head.appendChild(style);

    // 2. Ustvarimo HTML elemente
    const chatBubble = document.createElement('div');
    chatBubble.id = 'zupan-chat-bubble';
    // SVG ikona za oblaček
    chatBubble.innerHTML = `<svg viewBox="0 0 24 24"><path d="M20 2H4c-1.1 0-2 .9-2 2v18l4-4h14c1.1 0 2-.9 2-2V4c0-1.1-.9-2-2-2z"></path></svg>`;

    const chatWidget = document.createElement('div');
    chatWidget.id = 'zupan-chat-widget';
    chatWidget.innerHTML = `<iframe src="${chatUrl}"></iframe>`;

    document.body.appendChild(chatBubble);
    document.body.appendChild(chatWidget);

    // 3. Dodamo logiko za odpiranje in zapiranje
    chatBubble.addEventListener('click', () => {
        if (chatWidget.style.display === 'none' || chatWidget.style.display === '') {
            chatWidget.style.display = 'flex';
        } else {
            chatWidget.style.display = 'none';
        }
    });
})();