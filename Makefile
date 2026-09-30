HUGO := $(shell [ -x .bin/hugo ] && echo .bin/hugo || echo hugo)

# Preview locally, including drafts: http://localhost:1313
serve:
	$(HUGO) server -D

# Preview exactly what will be published (no drafts).
serve-public:
	$(HUGO) server

# make new name=some-slug
new:
	$(HUGO) new content notes/$(name).md

build:
	$(HUGO) --minify

.PHONY: serve serve-public new build
