-- Voice input for Claude Code. Add to ~/.wezterm.lua (dotfiles/.wezterm.lua).
--
-- CTRL+SHIFT+V starts recording; it stops on its own after ~1.8s of silence, or
-- press again to cut it short. The transcript is typed into the prompt for you
-- to review and send.
--
-- background_child_process is deliberate: SpawnCommandInNewWindow would steal
-- focus, and recording works better when the terminal you are talking at stays
-- in front of you.
local wezterm = require 'wezterm'

local VOICE = os.getenv('HOME') .. '/projects/github/unxmaal/localharness/voice/listen.sh'

-- Merge into your existing config.keys table.
return {
  {
    key = 'V',
    mods = 'CTRL|SHIFT',
    action = wezterm.action_callback(function(_, pane)
      wezterm.background_child_process({
        '/bin/bash', VOICE,
      })
      -- listen.sh reads WEZTERM_PANE from its own environment; pass it through
      -- explicitly so it knows where to type even when the pane changes.
      wezterm.log_info('voice: listening on pane ' .. tostring(pane:pane_id()))
    end),
  },
}
