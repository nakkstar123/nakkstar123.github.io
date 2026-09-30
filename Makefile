HUGO := $(shell [ -x .bin/hugo ] && echo .bin/hugo || echo hugo)

# Preview the site at http://localhost:1313 (reloads as you edit in Obsidian).
serve:
	$(HUGO) server

build:
	$(HUGO) --minify

.PHONY: serve build
